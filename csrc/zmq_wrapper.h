#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <utility>
#include "include/zmq.hpp"

// 默认地址（使用 IPC socket）
constexpr const char* DEFAULT_ADDRESS = "ipc:///tmp/zmq_default.sock";

class ZMQClient {
public:
    ZMQClient(const std::string& name = "",
              const std::string& address = DEFAULT_ADDRESS,
              int socket_type = ZMQ_DEALER);

    // 发送 unordered_map
    void send_map(const std::unordered_map<std::string, std::string>& dict);

    // 接收 unordered_map（非阻塞）
    std::optional<std::unordered_map<std::string, std::string>> recv_map_nonblock();

    // 接收 unordered_map（阻塞）
    std::unordered_map<std::string, std::string> recv_map();

    void close();

private:
    zmq::context_t context;
    zmq::socket_t socket;
    std::string name;

    static std::string serialize_map(const std::unordered_map<std::string, std::string>& dict);
    static std::unordered_map<std::string, std::string> deserialize_map(const std::string& s);
};

class ZMQServer {
public:
    ZMQServer(const std::string& address = DEFAULT_ADDRESS,
              int socket_type = ZMQ_ROUTER);

    // 接收一条消息（非阻塞），返回 (client_id, dict)
    std::optional<std::pair<std::string, std::unordered_map<std::string, std::string>>> recv_map_nonblock();

    // 接收所有消息（非阻塞）
    std::vector<std::pair<std::string, std::unordered_map<std::string, std::string>>> recv_all_map_nonblock();

    // 发送字典给指定客户端
    void send_map(const std::string& ident, const std::unordered_map<std::string, std::string>& dict);

    // 接收一条消息（阻塞）
    std::pair<std::string, std::unordered_map<std::string, std::string>> recv_map();

    void close();

private:
    zmq::context_t context;
    zmq::socket_t socket;

    static std::string serialize_map(const std::unordered_map<std::string, std::string>& dict);
    static std::unordered_map<std::string, std::string> deserialize_map(const std::string& s);
};
