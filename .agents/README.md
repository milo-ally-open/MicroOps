# Codex 项目级扩展

这里保存可提交、可随仓库分发的项目级配置索引和独立 skills。Codex 项目 skill 使用 `.agents/skills/<skill-name>/SKILL.md`；插件包位于仓库 `plugins/<plugin-name>/`，插件自身的 skills 放在插件包内的 `skills/`。本仓库维护本地 marketplace 索引 `.agents/plugins/marketplace.json`。

仓库根目录的 `.codex/` 被忽略，只留给当前机器的项目设置和个人工作副本。Codex 本地项目配置放在 `.codex/config.toml`；不要把唯一的 skill 或 plugin 源文件放在那里。当前示例本地配置启用 `opsfromscratch-reduce-sum@opsfromscratch`。新克隆可将 `config.example.toml` 复制为 `.codex/config.toml`，然后按需修改；该本地文件不会提交。

## 已有产物

- 独立 skill：`skills/add-operator/`，指导按现有工程约定添加新算子。
- 插件包：`../plugins/opsfromscratch-reduce-sum/`，内含 `reduce_sum` 学习 skill。
- Marketplace 名称：`opsfromscratch`，索引见 `plugins/marketplace.json`。

独立项目 skill 由 `.agents/skills/` 提供；可移植插件目录在仓库根部 `plugins/`。这两个位置是共享产物的来源。被忽略的 `.codex/skills/` 不会自动作为仓库 skill 被发现。

新增独立 skill 时，新建一个包含 `SKILL.md` 的子目录；新增插件时，在 `plugins/` 下建立独立插件包并登记到 marketplace。插件根目录使用可移植的 `plugin.json`，skills 按插件规范放在该插件根目录的 `skills/` 中。
