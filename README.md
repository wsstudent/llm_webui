# MiniCPM问答系统

这个项目是一个基于 MiniCPM 模型的问答系统，使用 Gradio 提供用户界面。用户可以上传文本或 PDF 文件，并进行查询，系统会返回相关的回答和文档链接。

## 功能

- 上传 PDF 和文本文件
- 对上传的文档进行处理和清理
- 使用 MiniCPM 模型生成回答
- 显示相关文档的引用信息
- 调整生成参数以优化回答

## 安装

请确保你已经安装了以下依赖项：

```bash
pip install langchain gradio torch transformers pydantic
