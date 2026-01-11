# fcitx5-dbus-bridge

## ChatGPT Threads

Branch · Fcitx5 输入法开发教程
https://chatgpt.com/share/6959e47d-8de0-8013-bcf6-b655c2f7e9bb

Fcitx5 输入法开发教程
https://chatgpt.com/share/6959e4aa-0d94-8013-827b-0488122c9a52

## How to build

编译依赖: 

一般你至少需要：

cmake ninja gcc/clang

fcitx5（含 headers / cmake config）
（如果 find_package(Fcitx5Core) 找不到，说明你缺 dev 文件；Arch 通常 fcitx5 包就带）

```
sudo pacman -S --needed base-devel cmake ninja pkgconf fcitx5 fcitx5-qt fcitx5-configtool
```

编译：

```
mkdir -p build
cmake -S . -B build -G Ninja -DCMAKE_INSTALL_PREFIX=/usr
cmake --build build
sudo cmake --install build
```

安装后：

```
/usr/lib/fcitx5/libdbusbridge.so
/usr/share/fcitx5/addon/dbusbridge.conf
```

让 fcitx5 重新加载:
1. 最粗暴：注销/登录一次
2. 或者（如果你是 systemd user service）：
systemctl --user restart fcitx5-daemon.service
3. 仅重启 fcitx：fcitx5-remote -r（很多环境有效）

启用 addon

打开 fcitx5-configtool → Addons → 勾上 dbusbridge（如果 OnDemand=False 通常会直接加载）。
但是在我的电脑上，无法勾选，安装上似乎就启用了

## 外部语音进程如何把文字“塞进输入框”

用 qdbus 测试（KDE 自带，最快）

```
qdbus org.fcitx.Fcitx5.DBusBridge \
  /org/fcitx/Fcitx5/DBusBridge \
  org.fcitx.Fcitx5.DBusBridge1.SendText \
  "你好，这是语音识别结果"
```

用 gdbus（更通用）

```
gdbus call --session \
  --dest org.fcitx.Fcitx5.DBusBridge \
  --object-path /org/fcitx/Fcitx5/DBusBridge \
  --method org.fcitx.Fcitx5.DBusBridge1.SendText \
  "hello from gdbus"
```

