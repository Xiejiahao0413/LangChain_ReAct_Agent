"""
yaml
K:V
配置文件处理
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import yaml
from utils.path_tool import get_abs_path

#从 config/rag.yml 文件加载RAG相关配置（如向量库路径、检索参数等），返回字典格式的配置信息。
def load_rag_config(config_path:str=get_abs_path("config/rag.yml"),encoding:str="utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.safe_load(f)
    

def load_chroma_config(config_path:str=get_abs_path("config/chroma.yml"),encoding:str="utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.safe_load(f)


def load_prompts_config(config_path:str=get_abs_path("config/prompts.yml"),encoding:str="utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.safe_load(f)
    

def load_agent_config(config_path:str=get_abs_path("config/agent.yml"),encoding:str="utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        return yaml.safe_load(f)
    

rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()


if __name__ == '__main__' :
    print(rag_conf["chat_model_name"])
