from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # 更新导入
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # 更新导入
from argparse import ArgumentParser
from langchain.llms.base import LLM
from typing import Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.prompts import PromptTemplate
from pydantic.v1 import Field
import re
import gradio as gr


class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本，去除中英文字符、数字及常见标点。

        参数:
        text (str): 需要清理的原始文本。

        返回:
        str: 清理后的文本。
        """
        pattern = r'[^\u4e00-\u9fa5A-Za-z0-9.,;!?()"\']'  # 修正为只保留有效字符
        cleaned_text = re.sub(pattern, "", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)  # 替换多余的空白字符
        return cleaned_text.strip()  # 去除首尾空白


class MiniCPM_LLM(LLM):
    tokenizer: Any = Field(default=None)
    model: Any = Field(default=None)
    backend: str = Field(default="hf")
    top_p: float = Field(default=0.9)
    temperature: float = Field(default=1.0)
    repetition_penalty: float = Field(default=1.0)
    max_dec_len: int = Field(default=1024)

    def __init__(self, model_path: str, device: str, backend: str, **kwargs):
        super().__init__(**kwargs)  # 使用kwargs传递所有Pydantic字段
        self.backend = backend
        self.tokenizer = None  # 根据需要初始化
        self.model = None  # 根据需要初始化

        if backend == "vllm":
            from vllm import LLM
            self.model = LLM(model=model_path, trust_remote_code=True, enforce_eager=True)
        else:
            # 初始化模型数据类型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.float16
            ).to(device).eval()

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(f"<用户>{prompt}", return_tensors="pt").to(torch.device("cuda:0"))
        with torch.autocast(device_type='cuda'):  # 使用自动混合精度
            generate_ids = self.model.generate(
                inputs.input_ids,
                top_p=self.top_p,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                max_length=self.max_dec_len,
                top_k=0,
            )
        responds = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        return responds

    @property
    def _llm_type(self) -> str:
        return "MiniCPM_LLM"


class DocumentHandler:
    @staticmethod
    def load_documents(file_paths: str) -> List[Any]:
        """
        加载文本和pdf文件中的字符串，并进行简单的清洗

        参数:
        file_paths (str or list): 传入的文件地址或者文件列表
        一个pdf会有多页

        返回:
        documents (list): 读取的文本列表
        """
        files_list = []
        if type(file_paths) == list:
            files_list = file_paths
        else:
            files_list = [file_paths]
        documents = []
        for file_path in files_list:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                raise ValueError("Unsupported file type")
            doc = loader.load()
            doc[0].page_content = TextProcessor.clean_text(doc[0].page_content)
            print("**********debug**********\n", "Loaded document content:", doc[0].page_content,
                  "\n**********debug**********")  # 添加调试打印
            documents.extend(doc)

        print("**********debug**********\n", documents,
              "\n**********debug**********")
        return documents  # 返回包含清理后的第一个文档的列表


class RAGChainCreator:
    @staticmethod
    def create_prompt_template() -> PromptTemplate:
        """
        创建自定义的prompt模板

        返回:
        PROMPT:自定义的prompt模板
        """
        custom_prompt_template = """请使用以下内容片段对问题进行最终回复，如果内容中没有提到的信息不要瞎猜，
        严格按照内容进行回答，不要编造答案，如果无法从内容中找到答案，请回答“片段中未提及，无法回答”，不要编造答案。
        Context:
        {context}

        Question: {question}
        FINAL ANSWER:"""
        return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    @staticmethod
    def create_rag_chain(llm, prompt) -> Any:
        """
        创建RAG链

        参数:
        llm: 语言模型
        prompt: 提示模板

        返回:
        qa: RAG链
        """
        return prompt | llm


class QueryAnalyzer:
    @staticmethod
    def analysis_links(docs: List[Any]) -> str:
        """
            分析文档链接并生成引用字符串。

            参数:
            docs (List[Any]): 读取的文档列表。

            返回:
            str: 相关文档引用字符串，格式为 "docname page content"。

            示例:
            >>> docs = [
            ...     {'source': '文档1', 'page': 1, 'content': '这是第一篇文档。'},
            ...     {'source': '文档2', 'page': 2, 'content': '这是第二篇文档。'}
            ... ]
            >>> analysis_links(docs)
            '文档1 page:1 \n\n这是第一篇文档。\n\n文档2 page:2 \n\n这是第二篇文档。'
        """
        if not docs:
            return "没有可分析的文档。"
        seen_sources = set()  # 用于存储已处理的文档源
        links_string = ""
        for doc in docs:
            source = doc.metadata["source"].split("/")[-1]
            if source not in seen_sources:  # 检查是否已经处理过
                seen_sources.add(source)
                links_string += f"{source} page:{doc.metadata['page']}\n\n{doc.page_content}\n\n"

        return links_string


class Utils:
    @staticmethod
    def clear_history():
        return []

    @staticmethod
    def reverse_last_round(chat_history):
        assert len(chat_history) >= 1, "历史记录为空。没有什么可以撤销的！！"
        return chat_history[:-1]


def regenerate(chatbot, top_p, temperature, repetition_penalty, max_dec_len):
    # 示例逻辑
    if len(chatbot) > 0:
        return chatbot[-1]
    return ""


class GradioInterface:
    def __init__(self, generator):
        self.generator = generator

    def launch(self, server_name: str, server_port: int):
        with gr.Blocks(theme="soft") as demo:
            gr.Markdown("# MiniCPM Gradio Demo")
            # 添加参数说明
            gr.Markdown("""
               ### 参数说明：
               - **top_p**: 核采样阈值，控制生成的多样性。值越小，输出越保守；值越大，输出越多样。
               - **temperature**: 控制生成文本的随机性。值越低，输出越确定；值越高，输出越随机。
               - **repetition_penalty**: 防止生成重复词汇的惩罚机制。值越高，重复可能性越低。
               - **max_dec_len**: 最大生成长度。控制生成的文本的最长字数。

               调整这些参数可以帮助""")

            with gr.Row():
                with gr.Column(scale=1):
                    top_p = gr.Slider(0, 1, value=0.8, step=0.1, label="top_p")
                    temperature = gr.Slider(0.1, 2.0, value=0.5, step=0.1, label="temperature")
                    repetition_penalty = gr.Slider(0.1, 2.0, value=1.1, step=0.1, label="repetition_penalty")
                    max_dec_len = gr.Slider(1, 1024, value=1024, step=1, label="max_dec_len")
                    link_content = gr.Textbox(label="link_content", lines=30, max_lines=40)

                with gr.Column(scale=5):
                    file_input = gr.File(label="upload_files", file_count="single")
                    # chatbot = gr.Chatbot(bubble_full_width=False, height=400)
                    final_anser = gr.Textbox(label="结合rag的输出", lines=5, max_lines=10)
                    user_input = gr.Textbox(label="用户", placeholder="在此输入您的查询！", lines=8)
                    with gr.Row():
                        submit = gr.Button("提交")
                        clear = gr.Button("清除")
                        # regen = gr.Button("重新生成")
                        reverse = gr.Button("撤销")

                    submit.click(self.generator.process_query,
                                 inputs=[file_input, user_input, top_p, temperature, repetition_penalty,
                                         max_dec_len],
                                 outputs=[final_anser, link_content])

                    # regen.click(regenerate,
                    #             inputs=[chatbot, top_p, temperature, repetition_penalty, max_dec_len],
                    #             outputs=[user_input, chatbot])
                    clear.click(Utils.clear_history, inputs=[], outputs=[final_anser])
                    reverse.click(Utils.reverse_last_round, inputs=[final_anser], outputs=[final_anser])

        demo.queue()
        demo.launch(server_name=server_name, server_port=server_port, show_error=True)


class MainApp:
    def __init__(self):
        """
        初始化MainApp类，加载模型并设置Gradio界面。
        """
        self.args = args  # 将args存储为实例属性
        self.documents = []  # 存储加载的文档
        self.vectorstore = None  # 向量数据库，用于检索相关文档
        self.rag_chain = None  # RAG链，用于生成答案
        self.embedding_models = self.load_embedding_models(self)  # 加载和嵌入模型
        # 初始化Gradio界面
        self.gradio_interface = GradioInterface(self)  # 将当前实例传入Gradio界面

    def load_llm_models(self, top_p: float, temperature: float, repetition_penalty: float, max_dec_len: int):
        """
        加载MiniCPM模型和嵌入模型。

        返回:
        llm: MiniCPM模型
        """
        llm_model = MiniCPM_LLM(
            model_path=self.args.cpm_model_path,
            device=self.args.cpm_device,
            backend=self.args.backend,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_dec_len=max_dec_len,
        )

        return llm_model

    @staticmethod
    def load_embedding_models(self):

        embedding_model = HuggingFaceBgeEmbeddings(
            model_name=self.args.encode_model,
            model_kwargs={"device": self.args.encode_model_device},
            encode_kwargs={
                "normalize_embeddings": True,
                "show_progress_bar": True,
                "convert_to_numpy": True,
                "batch_size": 8
            },
            query_instruction=self.args.query_instruction,
        )
        return embedding_model

    def process_query(self, file: str, query: str, top_p: float, temperature: float, repetition_penalty: float,
                      max_dec_len: int):
        print("**********debug**********\n", file, query, top_p, temperature, repetition_penalty, max_dec_len,
              "\n**********debug**********")
        """
        处理用户的查询,生成对话

        参数:
        file (str): 上传的文件路径，可以是文本文件或 PDF 文件。
        query (str): 用户输入的问题
        """
        # 检查文件类型
        if not isinstance(file, str):
            raise ValueError(f"无效的文件类型: {file}")

        # 如果上传的新文件与之前的文件不同，重新加载文档并更新向量存储和RAG链
        # getattr(self, 'exist_file', None) 用于获取上一次上传的文件路径，如果没有则默认为 None。
        if file != getattr(self, 'exist_file', None):
            """
            加载文档:调用 DocumentHandler 的 load_documents 方法来加载新的文档，并对其进行清洗和处理。
            创建向量存储:调用 embed_documents 方法来对文档进行分割和嵌入，创建向量数据库，用于后续的相似性搜索。
            创建RAG链:使用 RAGChainCreator 创建一个 RAG链，该链将结合语言模型（llm）和自定义的提示模板。
            isinstance(file, list) 检查 file 是否是一个列表。如果是，它就直接使用 file。
            如果 file 不是列表，那么它假定 file 是一个文件对象，因此使用 file.name 来获取文件的路径。
            """
            self.documents = DocumentHandler.load_documents(file if isinstance(file, list) else file.name)
            self.vectorstore = self.embed_documents(self.documents)
            self.rag_chain = RAGChainCreator.create_rag_chain(
                self.load_llm_models(top_p, temperature, repetition_penalty,
                                     max_dec_len), RAGChainCreator.create_prompt_template())

        # 在向量存储中对用户输入进行相似性搜索，获取相关文档
        docs = self.vectorstore.similarity_search(query, k=self.args.embed_top_k)  # embed_top_k 参数指定要返回的相似文档数量。
        all_links = QueryAnalyzer.analysis_links(docs)  # 对检索到的相关文档进行处理，生成包含文档引用信息的字符串。
        final_result = self.rag_chain.invoke(
            {"context": all_links[0:4], "question": query})  # 使用 RAG 链生成最终的答案，将上下文（all_links）和用户查询（query）作为输入
        # result = rag_chain({"input_documents": docs, "question": query}, return_only_outputs=False)

        # 解析生成的结果，仅返回最终的答案和相关文档链接。
        return final_result.split("FINAL ANSWER:")[-1], all_links

    def embed_documents(self, documents: List[Any]) -> Any:
        """
        对文档进行分割和嵌入，创建向量数据库。

        参数:
        documents (list): 读取的文本列表

        返回:
        vectorstore: 向量数据库
        """
        # 使用递归字符文本分割器进行文档分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.args.chunk_size,
                                                       chunk_overlap=self.args.chunk_overlap)
        texts = text_splitter.split_documents(documents)  # 分割文档
        print("Texts to embed:", texts)  # 添加调试打印
        if not texts:
            raise ValueError("没有有效的文本进行嵌入")
        return Chroma.from_documents(texts, self.embedding_models)  # 创建并返回向量存储

    def start_interface(self, server_name: str, server_port: int):
        self.gradio_interface.launch(server_name, server_port)


if __name__ == "__main__":
    parser = ArgumentParser()
    # ArgumentParser configuration goes here...
    """ 语言模型与检索模型的参数设置 """
    # 生成参数
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--repetition_penalty", type=float, default=1.02)

    # retriever参数设置
    parser.add_argument("--embed_top_k", type=int, default=5, help="召回几个最相似的文本")
    parser.add_argument("--chunk_size", type=int, default=256, help="文本切分时切分的长度")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="文本切分的重叠长度")
    parser.add_argument("--query_instruction", type=str, default="", help="召回时增加的前缀")

    parser.add_argument("--cpm_model_path", type=str, default="openbmb/MiniCPM-1B-sft-bf16",
                        help="MiniCPM模型路径或者huggingface id")
    parser.add_argument("--cpm_device", type=str, default="cuda:0", choices=["auto", "cuda:0"],
                        help="MiniCPM模型设备，默认为cuda:0")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"],
                        help="使用hf还是vllm后端，默认为hf")
    parser.add_argument("--encode_model", type=str, default="BAAI/bge-base-zh", help="用于召回编码的embedding模型")
    parser.add_argument("--encode_model_device", type=str, default="cuda:0", choices=["cpu", "cuda:0"],
                        help="嵌入模型设备，默认为cpu")
    parser.add_argument("--file_path", type=str, default="/root/ld/pull_request/rag/红楼梦.pdf",
                        help="需要检索的文本文件路径,gradio运行时无效")
    args = parser.parse_args()

    app = MainApp()
    app.start_interface(server_name="0.0.0.0", server_port=7860)
