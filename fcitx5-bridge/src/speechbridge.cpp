#include <fcitx/addonfactory.h>
#include <fcitx/addonmanager.h> 
#include <fcitx/addoninstance.h>
#include <fcitx/instance.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputcontextmanager.h>

#include <fcitx-utils/dbus/bus.h>
#include <fcitx-utils/dbus/objectvtable.h>
#include <fcitx-utils/log.h>

#include <dbus/dbus.h>

#include <string>
#include <memory>

namespace {

// 你外部语音进程要调用的 DBus 三件套
constexpr const char *kServiceName   = "org.fcitx.Fcitx5.SpeechBridge";
constexpr const char *kObjectPath    = "/org/fcitx/Fcitx5/SpeechBridge";
constexpr const char *kInterfaceName = "org.fcitx.Fcitx5.SpeechBridge1";

} // namespace

class SpeechBridgeModule;

// DBus 对象：提供方法 SendText(string) -> bool
class SpeechBridgeDBusObject : public fcitx::dbus::ObjectVTable<SpeechBridgeDBusObject> {
public:
    explicit SpeechBridgeDBusObject(SpeechBridgeModule *owner) : owner_(owner) {}

    // DBus: SendText(s) -> b
    bool SendText(const std::string &text);

    // 把成员函数注册为 DBus method：
    //   method name = "SendText"
    //   args signature = "s" (string)
    //   return signature = "b" (boolean)
    FCITX_OBJECT_VTABLE_METHOD(SendText, "SendText", "s", "b");

private:
    SpeechBridgeModule *owner_;
    void logSenderMetadata() const;
};

class SpeechBridgeModule : public fcitx::AddonInstance {
public:
    explicit SpeechBridgeModule(fcitx::Instance *instance)
        : instance_(instance),
          bus_(fcitx::dbus::BusType::Session),
          dbusObject_(this) {

        // 把 DBus bus 挂到 fcitx 的 event loop 上（同线程/同循环，最省事）
        bus_.attachEventLoop(&instance_->eventLoop());

        if (!bus_.isOpen()) {
            FCITX_ERROR() << "SpeechBridge: DBus session bus is not open.";
            return;
        }

        // 申请服务名（可用 ReplaceExisting 方便你重启 fcitx5 测试）
        if (!bus_.requestName(
                kServiceName,
                fcitx::Flags<fcitx::dbus::RequestNameFlag>{fcitx::dbus::RequestNameFlag::ReplaceExisting})) {
            FCITX_ERROR() << "SpeechBridge: failed to request dbus name: " << kServiceName;
            return;
        }

        // 注册对象与接口
        if (!bus_.addObjectVTable(kObjectPath, kInterfaceName, dbusObject_)) {
            FCITX_ERROR() << "SpeechBridge: failed to addObjectVTable at " << kObjectPath;
            bus_.releaseName(kServiceName);
            return;
        }

        FCITX_INFO() << "SpeechBridge: ready on DBus:"
                     << " name=" << kServiceName
                     << " path=" << kObjectPath
                     << " iface=" << kInterfaceName;
    }

    ~SpeechBridgeModule() override {
        // 安全清理
        bus_.releaseName(kServiceName);
        bus_.detachEventLoop();
    }

    // 给 DBus 对象调用：把 text 提交到当前输入上下文
    bool commitTextToClient(const std::string &text) {
        if (text.empty()) {
            return false;
        }

        // 优先 lastFocused；没有就退化到 mostRecent（更抗“focus out”）
        fcitx::InputContext *ic = instance_->inputContextManager().lastFocusedInputContext();
        if (!ic) {
            ic = instance_->inputContextManager().mostRecentInputContext();
        }

        if (!ic) {
            FCITX_WARN() << "SpeechBridge: no input context available, drop text.";
            return false;
        }

        FCITX_INFO() << "SpeechBridge: commit text, len=" << text.size();
        ic->commitString(text);
        return true;
    }

private:
    friend class SpeechBridgeDBusObject;

    fcitx::Instance *instance_;
    fcitx::dbus::Bus bus_;
    SpeechBridgeDBusObject dbusObject_;
};

void SpeechBridgeDBusObject::logSenderMetadata() const {
    auto message = this->currentMessage();
    if (!message) {
        FCITX_WARN() << "SpeechBridge: SendText invoked with no current message";
        return;
    }

    // High-level metadata (always available)
    std::string sender = message->sender();
    std::string method = message->member();
    std::string interface = message->interface();
    std::string path = message->path();

    // Low-level metadata
    uint32_t serial = 0;

    auto dbusMsg = static_cast<DBusMessage*>(message->nativeHandle());

    if (dbusMsg) {
        serial = dbus_message_get_serial(dbusMsg);
    }

    // Note: Getting PID/UID requires blocking DBus calls which would deadlock
    // when called from within a message handler. For security/audit purposes,
    // the sender bus name can be used to query credentials separately if needed.

    // Log all metadata in single line
    FCITX_INFO() << "SpeechBridge SendText: sender=" << sender
                 << " serial=" << serial
                 << " method=" << method
                 << " interface=" << interface
                 << " path=" << path;
}

bool SpeechBridgeDBusObject::SendText(const std::string &text) {
    // DBus method 入口
    if (!owner_) {
        return false;
    }

    logSenderMetadata();

    return owner_->commitTextToClient(text);
}

// 工厂
class SpeechBridgeFactory : public fcitx::AddonFactory {
public:
    fcitx::AddonInstance *create(fcitx::AddonManager *manager) override {
        auto *instance = manager->instance();
        return new SpeechBridgeModule(instance);
    }
};


FCITX_ADDON_FACTORY(SpeechBridgeFactory);
