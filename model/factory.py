"""
模型工厂模式:用于统一创建Chat模型和Embedding模型
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
# from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:      
        pass
    """
    定义抽象方法 generator()，子类必须实现
    返回类型：Embeddings（向量模型）或 BaseChatModel（对话模型）
    ABC 表示抽象类，不能直接实例化
    """


class ChatModelFactory(BaseModelFactory):
    def generator(self)->Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf["chat_model_name"])


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()

embed_model = EmbeddingsFactory().generator()

"""
创建工厂对象并调用 generator() 获取模型实例
chat_model：可直接调用的对话模型
embed_model：用于文本向量化的模型
"""


"""
设计模式优点
解耦：业务代码不直接依赖具体模型类
扩展性：新增模型只需添加新工厂类
统一管理：所有模型配置集中在 rag.yml
"""