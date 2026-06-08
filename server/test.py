import asyncio

class MyClass:
    async def _main_event_loop(self):
        cnt = 1
        while True:
            print(f'正在循环{cnt}')
            print(f'正在循环xx')
            await asyncio.sleep(1)
            cnt += 1
        print("循环结束")


if __name__ == "__main__":
    mc = MyClass()
    # 启动异步函数
    asyncio.run(mc._main_event_loop())
