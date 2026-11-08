"""Small, thread-safe implementation of the upstream Aqua coordinator API.

Offers represent spare GPU bytes; allocations are indivisible Aqua tensors.
Reclaim quarantines an offer until consumers have migrated and freed every
allocation. Capacity is reduced only after these acknowledgments.
"""

import argparse
from dataclasses import dataclass, asdict
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class LedgerError(ValueError):
    pass


@dataclass
class Offer:
    address: str
    capacity: int
    allocated: int = 0
    reclaiming: bool = False


class Ledger:
    def __init__(self):
        self.lock = threading.RLock()
        self.offers, self.mappings, self.allocations = {}, {}, {}
        self.events = []

    @staticmethod
    def integer(value, minimum=0):
        if type(value) is not int or value < minimum:
            raise LedgerError(f"Expected integer >= {minimum}")
        return value

    def log(self, event, **data):
        self.events.append(dict(event=event, time_ns=time.monotonic_ns(), **data))
        self.events = self.events[-10000:]

    def offer(self, ident, address, size):
        with self.lock:
            self.integer(ident)
            self.integer(size)
            if ident in self.offers or address != f"cuda:{ident}":
                raise LedgerError(
                    "Duplicate provider or inconsistent physical CUDA address"
                )
            self.offers[ident] = Offer(address, size)
            self.log("offer", producer=ident, capacity=size)
            return asdict(self.offers[ident])

    def resize(self, ident, delta):
        with self.lock:
            store = self.offers[ident]
            if type(delta) is not int or store.capacity + delta < store.allocated:
                raise LedgerError(
                    "Capacity cannot become negative or cover live allocations"
                )
            old = store.capacity
            store.capacity += delta
            self.log("offer_resized", producer=ident, old=old, capacity=store.capacity)
            return asdict(store)

    def map(self, consumer, producer):
        with self.lock:
            self.integer(consumer)
            self.integer(producer)
            if (
                consumer == producer
                or consumer in self.mappings
                or producer not in self.offers
            ):
                raise LedgerError("Invalid or duplicate producer/consumer mapping")
            self.mappings[consumer] = producer
            self.log("mapping", consumer=consumer, producer=producer)

    def allocate(self, consumer, size):
        with self.lock:
            self.integer(size, 1)
            producer = self.mappings.get(consumer)
            store = self.offers.get(producer)
            if (
                store is None
                or store.reclaiming
                or store.capacity - store.allocated < size
            ):
                self.log("dram_fallback", consumer=consumer, size=size)
                return None
            ident = str(uuid.uuid4())
            record = dict(
                size=size,
                address=store.address,
                store_id=producer,
                allocation_id=ident,
                consumer=consumer,
            )
            self.allocations[ident] = record
            store.allocated += size
            self.log("allocation", **record)
            return record.copy()

    def free(self, ident, consumer):
        with self.lock:
            item = self.allocations.get(ident)
            if item is None:
                return dict(freed=False, already_released=True)
            if item["consumer"] != consumer:
                raise LedgerError("Allocation is owned by another consumer")
            self.offers[item["store_id"]].allocated -= item["size"]
            del self.allocations[ident]
            self.log(
                "allocation_freed",
                allocation_id=ident,
                size=item["size"],
                producer=item["store_id"],
            )
            return dict(freed=True)

    def reclaim(self, producer):
        with self.lock:
            self.offers[producer].reclaiming = True
            self.log("reclaim_requested", producer=producer)
            return dict(added=True)

    def reclaim_status(self, producer):
        with self.lock:
            store = self.offers[producer]
            return dict(
                capacity=store.capacity,
                available=store.capacity - store.allocated,
                can_reclaim=store.allocated == 0,
            )

    def finish_reclaim(self, producer):
        with self.lock:
            store = self.offers[producer]
            if store.allocated:
                raise LedgerError(
                    "Consumer migration acknowledgments are still pending"
                )
            store.reclaiming = False
            self.log("reclaim_completed", producer=producer, capacity=store.capacity)
            return dict(added=True)

    def responsive(self, identifiers, consumer):
        with self.lock:
            return [
                ident
                for ident in identifiers
                if ident in self.allocations
                and self.allocations[ident]["consumer"] == consumer
                and self.offers[self.allocations[ident]["store_id"]].reclaiming
            ]

    def unmap(self, consumer):
        with self.lock:
            if any(a["consumer"] == consumer for a in self.allocations.values()):
                raise LedgerError("Cannot unmap a consumer with live allocations")
            self.mappings.pop(consumer, None)

    def unlease(self, producer):
        with self.lock:
            if self.offers[producer].allocated:
                raise LedgerError("Cannot remove a live offer")
            self.offers.pop(producer)
            self.mappings = {c: p for c, p in self.mappings.items() if p != producer}
            return dict(deleted=True)

    def snapshot(self):
        with self.lock:
            for ident, offer in self.offers.items():
                assert offer.allocated == sum(
                    a["size"]
                    for a in self.allocations.values()
                    if a["store_id"] == ident
                )
                assert 0 <= offer.allocated <= offer.capacity
            return dict(
                providers={str(i): asdict(s) for i, s in self.offers.items()},
                allocations=list(self.allocations.values()),
                mappings=self.mappings.copy(),
                events=self.events.copy(),
            )


def make_app(ledger=None):
    ledger = ledger or Ledger()
    app = FastAPI()

    @app.exception_handler(LedgerError)
    async def invalid(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"error": "Unknown provider/allocation"}, status_code=404)

    @app.get("/health")
    def health():
        return {"status": "ready"}

    @app.get("/status")
    def status():
        return ledger.snapshot()

    @app.post("/lease")
    def offer(data: dict):
        return ledger.offer(data["id"], data["address"], data["size"])

    @app.put("/lease")
    def resize(data: dict):
        return ledger.resize(data["id"], data["size"])

    @app.post("/mapping")
    def mapping(data: dict):
        ledger.map(data["consumer"], data["producer"])
        return {"mapped": True}

    @app.delete("/mapping")
    def unmap(data: dict):
        ledger.unmap(data["consumer"])
        return {"deleted": True}

    @app.delete("/unlease")
    def unlease(data: dict):
        return ledger.unlease(data["id"])

    @app.post("/nv_allocate")
    def allocate(data: dict, request: Request):
        return ledger.allocate(
            int(request.headers.get("source-gpu", "-1")), data["memory"]
        )

    @app.delete("/nv_free")
    def free(data: dict, request: Request):
        return ledger.free(
            data["allocation_id"], int(request.headers.get("source-gpu", "-1"))
        )

    @app.post("/reclaim_request")
    def reclaim(data: dict):
        return ledger.reclaim(data["id"])

    @app.delete("/reclaim_request")
    def finish(data: dict):
        return ledger.finish_reclaim(data["id"])

    @app.post("/reclaim_status")
    def reclaim_status(data: dict):
        return ledger.reclaim_status(data["id"])

    @app.post("/responsive_reclaim")
    def responsive(data: dict, request: Request):
        return ledger.responsive(
            data["allocations"], int(request.headers.get("source-gpu", "-1"))
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18101)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(make_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
