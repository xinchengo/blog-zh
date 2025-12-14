---
date: 2025-02-08
---

# Home Assistant 之开发笔记

!!! warning "本文正在编写中，并且可能长时间处于这一状态"

写这篇文章的原因是，家里有很多智能家居设备，但是智能家居设备之间组合的方式比较死板，不能自由地加以组合。看到小米官方在官方界面上发布了 [Home Assistant 的集成](https://github.com/XiaoMi/ha_xiaomi_home/tree/main)，我关注到了 [Home Assistant](https://www.home-assistant.io/) 这个开源项目——事实上，按[开发者的数量来算](https://github.blog/news-insights/octoverse/octoverse-2024/#the-state-of-open-source)，他是整个 2024 年 Github 上最受欢迎的项目。

<!-- more -->

既然有这么好的现成项目，与其去重复造轮子，不如去直接使用这个人类智慧结晶。遗憾的是，这个应用似乎主要是为欧美的使用者开发的——比如，它的安装过程需要连接 `github.com` 和 `ghcr.io`，对于中国国内的使用者很不友好。使用了小米 API 以后，本地化做得也比较一般。天气、地图、智能助理使用的都是国外 API。
