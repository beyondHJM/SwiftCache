// scheduler_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // 支持 Python list <-> std::vector
#include <pybind11/functional.h>   // 如果有 std::function
#include <pybind11/stl_bind.h>     // 如果要支持绑定 STL 容器引用
#include <any>
#include <optional>

#include "scheduler.h"
#include "sequence.h"              // 加上你的 Sequence 类定义

namespace py = pybind11;

PYBIND11_MODULE(c_scheduler, m) {
    m.doc() = "Scheduler & Sequence C++ bindings for nanovllm";

    // ------------------ SchedulingType 枚举 ------------------
    // py::enum_<SchedulingType>(m, "SchedulingType")
    //     .value("PREFILL", SchedulingType::PREFILL)
    //     .value("DECODING", SchedulingType::DECODING);

    // ------------------ Scheduler 类绑定 ------------------
    py::class_<Scheduler>(m, "Scheduler")
        .def(py::init<py::object>(), py::arg("config"),
             "接收 Python Config 对象来初始化 Scheduler")
        .def("check_kvcache_change", &Scheduler::check_kvcache_change)
        .def("is_finished", &Scheduler::is_finished)
        .def("add", &Scheduler::add, py::arg("seq"))
        .def("schedule", &Scheduler::schedule)
        .def("preempt", &Scheduler::preempt)
        .def("postprocess", &Scheduler::postprocess,
             py::arg("seqs"), py::arg("token_ids"))
        .def("create_add",
                &Scheduler::create_add,
                py::arg("token_ids"),
                py::arg("sampling_params"),
                py::arg("input_embeds") = std::nullopt,
                py::arg("seq_id") = std::nullopt,
                "在C++中创建Sequence对象并加入队列");
    // ------------------ SequenceStatus 枚举 ------------------
    py::enum_<SequenceStatus>(m, "SequenceStatus")
        .value("WAITING", SequenceStatus::WAITING)
        .value("FINISHED", SequenceStatus::FINISHED)
        .value("RUNNING", SequenceStatus::RUNNING);

    // ------------------ Sequence 类绑定 ------------------
    py::class_<Sequence>(m, "Sequence")
        .def(py::init<
            const std::vector<int>&,
            py::object,
            std::optional<std::any>,
            std::optional<int>
        >(),
        py::arg("token_ids"),
        py::arg("sampling_params"),
        py::arg("input_embeds") = std::nullopt,
        py::arg("seq_id") = std::nullopt
        )
        // 成员函数绑定
        .def("size", &Sequence::size)
        .def("__getitem__", &Sequence::operator[])
        // .def("is_finished", &Sequence::is_finished)
        .def_property_readonly("is_finished", &Sequence::is_finished)
        .def("num_completion_tokens", &Sequence::num_completion_tokens)
        .def("prompt_token_ids", &Sequence::prompt_token_ids)
        // .def("completion_token_ids", &Sequence::completion_token_ids)
        .def_property_readonly("completion_token_ids", &Sequence::completion_token_ids)
        .def("num_cached_blocks", &Sequence::num_cached_blocks)
        .def("num_blocks", &Sequence::num_blocks)
        .def("__len__", &Sequence::length)
        .def("last_block_num_tokens", &Sequence::last_block_num_tokens)
        .def("block", &Sequence::block)
        .def("append_token", &Sequence::append_token)
        .def("get_state", &Sequence::get_state)
        .def("set_state", &Sequence::set_state)
        .def("time_usage_append",
            &Sequence::time_usage_append,
            py::arg("t"))
        // 公共成员变量暴露（如果需要 Python 能直接访问）
        .def_readwrite("seq_id", &Sequence::seq_id)
        .def_readwrite("status", &Sequence::status)
        .def_readwrite("input_embeds", &Sequence::input_embeds)
        .def_readwrite("token_ids", &Sequence::token_ids)
        .def_readwrite("last_token", &Sequence::last_token)
        .def_readwrite("num_tokens", &Sequence::num_tokens)
        .def_readwrite("num_prompt_tokens", &Sequence::num_prompt_tokens)
        .def_readwrite("num_cached_tokens", &Sequence::num_cached_tokens)
        .def_readwrite("temperature", &Sequence::temperature)
        .def_readwrite("max_tokens", &Sequence::max_tokens)
        .def_readwrite("ignore_eos", &Sequence::ignore_eos)
        .def_readwrite("time_usage", &Sequence::time_usage)
        .def_readwrite("extra_info", &Sequence::extra_info)
        .def_readwrite("block_table", &Sequence::block_table);
}
