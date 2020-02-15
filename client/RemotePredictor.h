#pragma once
#include <string>
#include <memory>


struct RemoteInfo {
    int init(const std::string& ipaddr, int port);
    std::string ipaddr;
    int port;
};

class RemotePredictor {
public:
    RemotePredictor();
    ~RemotePredictor();
    int init(const RemoteInfo& ri, const std::string& id);
    int predict(int w, int h, int c, void* data, int len, std::function<void()> callback);
    int release();
protected:
    struct Impl;
    Impl* impl_;
};
