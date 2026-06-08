#include "zmq_wrapper.h"
#include "include/json.hpp"      // 引入 nlohmann JSON
#include <filesystem>

// ====== 序列化和反序列化工具函数 (用 JSON 库) ======
std::string ZMQClient::serialize_map(const std::unordered_map<std::string, std::string>& dict) {
    nlohmann::json j(dict);  // 构造 JSON，直接用 unordered_map 初始化
    return j.dump();         // 转成 string（无缩进）
}

std::unordered_map<std::string, std::string> ZMQClient::deserialize_map(const std::string& s) {
    std::unordered_map<std::string, std::string> dict;
    auto j = nlohmann::json::parse(s);
    for (auto& [key, val] : j.items()) {
        dict[key] = val.get<std::string>();
    }
    return dict;
}

std::string ZMQServer::serialize_map(const std::unordered_map<std::string, std::string>& dict) {
    nlohmann::json j(dict);
    return j.dump();
}

std::unordered_map<std::string, std::string> ZMQServer::deserialize_map(const std::string& s) {
    std::unordered_map<std::string, std::string> dict;
    auto j = nlohmann::json::parse(s);
    for (auto& [key, val] : j.items()) {
        dict[key] = val.get<std::string>();
    }
    return dict;
}

// ====== ZMQClient ======
ZMQClient::ZMQClient(const std::string& name_,
                     const std::string& address,
                     int socket_type)
    : context(1), socket(context, socket_type), name(name_)
{
    socket.setsockopt(ZMQ_IDENTITY, name.data(), name.size());
    socket.connect(address);
}

void ZMQClient::send_map(const std::unordered_map<std::string, std::string>& dict) {
    auto data_str = serialize_map(dict);
    zmq::message_t msg(data_str.begin(), data_str.end());
    socket.send(msg, zmq::send_flags::none);
}

std::optional<std::unordered_map<std::string, std::string>> ZMQClient::recv_map_nonblock() {
    zmq::message_t msg;
    auto res = socket.recv(msg, zmq::recv_flags::dontwait);
    if (!res) return std::nullopt;
    std::string s(static_cast<char*>(msg.data()), msg.size());
    return deserialize_map(s);
}

std::unordered_map<std::string, std::string> ZMQClient::recv_map() {
    zmq::message_t msg;
    socket.recv(msg, zmq::recv_flags::none);
    std::string s(static_cast<char*>(msg.data()), msg.size());
    return deserialize_map(s);
}

void ZMQClient::close() {
    socket.close();
    context.close();
}

// ====== ZMQServer ======
ZMQServer::ZMQServer(const std::string& address, int socket_type)
    : context(1), socket(context, socket_type)
{
    if (address.rfind("ipc://", 0) == 0) {
        std::string path = address.substr(std::string("ipc://").size());
        if (std::filesystem::exists(path)) {
            std::filesystem::remove(path);
        }
    }
    socket.bind(address);
}

std::optional<std::pair<std::string, std::unordered_map<std::string, std::string>>> 
ZMQServer::recv_map_nonblock() {
    zmq::message_t ident_msg;
    zmq::message_t data_msg;
    auto res = socket.recv(ident_msg, zmq::recv_flags::dontwait);
    if (!res) return std::nullopt;
    socket.recv(data_msg, zmq::recv_flags::none);

    std::string ident(static_cast<char*>(ident_msg.data()), ident_msg.size());
    std::string data_str(static_cast<char*>(data_msg.data()), data_msg.size());
    return std::make_pair(ident, deserialize_map(data_str));
}

std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> 
ZMQServer::recv_all_map_nonblock() {
    std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> msgs;
    while (true) {
        zmq::message_t ident_msg;
        zmq::message_t data_msg;
        auto res = socket.recv(ident_msg, zmq::recv_flags::dontwait);
        if (!res) break;
        socket.recv(data_msg, zmq::recv_flags::none);

        std::string ident(static_cast<char*>(ident_msg.data()), ident_msg.size());
        std::string data_str(static_cast<char*>(data_msg.data()), data_msg.size());
        msgs.emplace_back(ident, deserialize_map(data_str));
    }
    return msgs;
}

void ZMQServer::send_map(const std::string& ident,
                         const std::unordered_map<std::string, std::string>& dict) {
    auto data_str = serialize_map(dict);
    zmq::message_t ident_msg(ident.begin(), ident.end());
    zmq::message_t data_msg(data_str.begin(), data_str.end());
    socket.send(ident_msg, zmq::send_flags::sndmore);
    socket.send(data_msg, zmq::send_flags::none);
}

std::pair<std::string, std::unordered_map<std::string, std::string>> ZMQServer::recv_map() {
    zmq::message_t ident_msg;
    zmq::message_t data_msg;
    socket.recv(ident_msg, zmq::recv_flags::none);
    socket.recv(data_msg, zmq::recv_flags::none);

    std::string ident(static_cast<char*>(ident_msg.data()), ident_msg.size());
    std::string data_str(static_cast<char*>(data_msg.data()), data_msg.size());
    return {ident, deserialize_map(data_str)};
}

void ZMQServer::close() {
    socket.close();
    context.close();
}
