# Arch Linux 安装与配置

这篇文章记录了我安装 Arch Linux 的经过，以及维护 Arch Linux 的一些小技巧，以便之后遇到问题时回溯我最初安装的经过。

安装 Arch Linux 其实并不算困难，因为 [Arch Linux 的官网](https://archlinux.org/)上有相当详细的[安装教程](https://wiki.archlinux.org/title/Installation_guide)，何况官方的安装镜像里有一个自带的“图形化”安装器`archinstall`。

## 安装系统

官网上的[教程](https://wiki.archlinux.org/title/Installation_guide)已经相当详细了，按照该提示下载好镜像，并烧录到 U 盘中。

我下面的安装方法使用了 Arch 官方提供的 [`archinstall`](https://wiki.archlinux.org/title/Archinstall)，对于初学者配置一个**供自己使用的操作系统**（而不是作为练习），建议使用安装脚本而不是按照教程手动安装，因为你可能会忽略掉安装系统中需要注意的种种细节。

安装时注意以下几点：

- 先用 `reflector --country=china > /etc/pacman.d/mirrorlist` 把 `mirrorlist` 更换成国内镜像，之后在 `archinstall` 界面上**不要管** `Mirror Region` 的配置。因为如果你动了这个设置，之后安装的过程中 `reflector` 就可能破坏原来的镜像设置，导致下载速度变慢（原因见 [这个 Github 帖子](https://github.com/archlinux/archinstall/issues/1429#issuecomment-1221929472)）。
- 文件系统选 Btrfs，一路默认设置就行了（但是可以 Disable Copy-on-Write）

这样选好桌面环境，安装完重新启动，不出差错的话就可以直接进入带图形界面的系统了。

## 安装工具

先用 `sudo pacman -Syu` 安装好 `firefox` 浏览器。

### 中文支持

先更新 `locale`：

   - `sudo vim /etc/locale.gen`，解除 `zh-CN.UTF-8 UTF-8` 前的注释
   - `sudo locale-gen`

再安装中文字体：
   
```bash
sudo pacman -Syu adobe-source-han-sans-cn-fonts\
    adobe-source-han-serif-cn-fonts\ 
    noto-fonts-cjk wqy-microhei\
    wqy-microhei-lite wqy-bitmapfont\
    wqy-zenhei ttf-arphic-ukai\
    ttf-arphic-uming
```

Fcitx5 输入法：

```bash
pacman -Syu fcitx5-im
sudo pacman -Syu fcitx5-im
sudo pacman -Syu fcitx5-im fcitx5-chinese-addons
```

### 安装 Windows 字体

### 安装必要的应用

用 [AUR 辅助工具](https://wiki.archlinux.org/title/AUR_helpers) [`yay`](https://github.com/Jguer/yay) 管理 AUR 包：

```bash
sudo pacman -Syu git
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si
```

之后就可以用 `yay -Syu <package name>` 安装想要安装的应用了。

### 配置系统维护工具

注意，系统维护是相当重要的！比如如果你在日后使用 Arch 的时候不小心遇到了难以恢复的问题，想要回滚到之前的版本，如果你之前已经**做好了系统快照**，解决问题就会方便很多。再比如说，如果你想在一台新的电脑上安装 Arch，但想保留你自己的配置，如果你使用了合适的方法做了**配置管理**，这个流程也会方便很多。

具体的系统维护见 [官方 Wiki](https://wiki.archlinux.org/title/System_maintenance)，但这里我也给出我自己的配置。

### 使用 Snapper 和 Btrfs 镜像制作系统快照

[Snapper](https://wiki.archlinux.org/title/Snapper) 是一款系统快照管理工具，支持制作快照，恢复快照和自动制作快照，安装方法如下：

```bash
sudo pacman -Syu cronie # 先安装 cron（定时执行工具）
sudo systemctl enable --now cronie # 启动 cron 服务

sudo pacman -Syu snapper
```

之后就是配置快照存储的目录，注意，如果你使用了 `archinstall` 推荐的 `Btrfs` 分区布局，需要用以下方法才能配置（见[官方文档 中的 Note](https://wiki.archlinux.org/title/Snapper#Creating_a_new_configuration)）：

```bash
sudo umount /.snapshots # 解除对 /.snapshots 的挂载
sudo rm -rf /.snapshots

sudo snapper -c root create-config / # 生成快照配置

sudo btrfs subvolume delete /.snapshots
sudo mount -o subvol=@.snapshots /dev/<sdX> /.snapshots/ # <sdX> 应换成你的 Btrfs 分区
```

此时自动快照就已经配置好了，如果要修改自动快照的保存数量（多少个每小时快照，每日快照……），那么在 `sudo vim /etc/snapper/configs/root` 中修改即可。

常用的管理命令有（更多的见[官方文档](https://wiki.archlinux.org/title/Snapper)）：
- `snapper -c <config> list`：列举按照 `config` 配置备份的镜像。
- `snapper -c <config> create --description <desc>`：手动创建一个信息为 `desc` 的快照，注意这个快照不会被自动删除。
- `snapper -c <config> create --command <cmd>`：执行 `cmd` 这个重要命令，并在命令执行前和执行后创建一对快照。
- `snapper -c <config> delete N`：删除 `N` 号快照。

### 用 Aconfmgr 管理配置文件

先 `yay -Syu aconfmgr-git` 安装 [Aconfmgr](https://github.com/CyberShadow/aconfmgr)。

之后配置排除清单（`~/.config/aconfmgr/10-ignores.sh`），因为有些目录下的文件没有必要还原，下面是一个示范的排除清单：

```sh
# Btrfs snapshots
IgnorePath '/.snapshots/*'
# Boot binaries
IgnorePath '/boot/*.img'
IgnorePath '/boot/*/*.EFI'
IgnorePath '/boot/*/*.efi'
IgnorePath '/boot/vmlin*'
# Certificate databases
IgnorePath '/etc/ca-certificates/extracted/*'
IgnorePath '/etc/ssl/certs/*'
IgnorePath '/etc/pacman.d/gnupg/*'
# Cache and generated files
IgnorePath '/etc/*.cache'
IgnorePath '/etc/*.gen'
# Password files
IgnorePath '/etc/*shadow*'
IgnorePath '/usr/share/*'
# Configuration database
IgnorePath '/etc/dconf'
# Mount files
IgnorePath '*/.updated'
IgnorePath '*/lost+found/*'
# Opt packages (check that they don't include config)
IgnorePath '/opt/*'
# Binary libraries
IgnorePath '/usr/lib/*'
# Local binaries
IgnorePath '/usr/local/include/*'
IgnorePath '/usr/local/lib/*'
IgnorePath '/usr/local/share/applications/mimeinfo.cache'
# Var databases, logs, swap and temp files
IgnorePath '/var/db/sudo'
IgnorePath '/var/lib/*'
IgnorePath '/var/log/*'
IgnorePath '/var/swap*'
IgnorePath '/var/tmp/*'
```

`aconfmgr save` 的作用就是将系统中所有变化的配置文件存储到 `~/.config/aconfmgr/`，而 `aconfmgr apply` 的作用正好相反，将 `~/.config/aconfmgr` 中的文件恢复到系统中。

写好排除清单后，我们就可以执行 `aconfmgr save` 制作配置文件的备份了。注意 `aconfmgr` 并不支持回滚，需要我们用 Git 维护 `~/.config/aconfmgr` 这个目录，手动 Commit，才能支持回滚功能。

### 配置 systemd-boot 和 Netboot.xyz

关于 Windows 双系统见[官方教程](https://wiki.archlinux.org/title/Systemd-boot#Boot_from_another_disk)（若 Windows 在另外一个硬盘上则需要此配置）

若要支持[记住上次进入的系统](https://wiki.archlinux.org/title/Systemd-boot#Remember_last_entry)，方法是在 `/boot/loader/loader.conf` 中加入 `default @saved`。

[Netboot.xyz](https://github.com/netbootxyz/netboot.xyz) 是一个实用工具，支持在启动的时候加载各种系统，主要用于系统救援。使用的方法看了 [Arch Linux 对于 `systemd-boot` 的介绍](https://wiki.archlinux.org/title/Systemd-boot#Configuration)应该不难学会，但是下面还是列出具体方法：

```bash
# 在官网上下载最新的 netboot.xyz 镜像
wget https://boot.netboot.xyz/ipxe/netboot.xyz.efi
# 将镜像拷贝到 EFI 文件夹中
sudo mkdir /boot/EFI/network.xyz
sudo cp netboot.xyz.efi /boot/EFI/netboot.xyz/
# 新建 systemd-boot 配置文件
sudo vim /boot/loader/entries/netboot_xyz.conf
```

文件内容如下：

```conf
title   netboot.xyz Utilities
efi     /EFI/netboot.xyz/netboot.xyz.efi
```

写入完成以后，使用 `bootctl list` 检查新建的 `netboot.xyz Utilities` 是否已经成为一项新的启动配置。

## 其他维护

[维护已安装的程序列表](https://wiki.archlinux.org/title/Pacman/Tips_and_tricks#List_of_installed_packages)：`pacman -Qqen > pkglist.txt && pacman -Qqem > foreignpkglist.txt
`