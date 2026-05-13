"""
文件处理工具
"""

import os 
import hashlib
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


#计算文件的MD5哈希值，用于文件唯一标识和完整性校验
def get_file_md5_hex(filepath:str):      #定义函数，接受文件路径，返回MD5十六进制字符串
    if not os.path.exists(filepath):     #文件不存在则记录错误并返回
        logger.error(f"[md5计算]文件{filepath}不存在")
        return
    
    if not os.path.isfile(filepath):     #路径不是文件则记录错误并返回
        logger.error(f'[md5计算]路径{filepath}不是文件')
        return
    
    md5_obj = hashlib.md5()         #创建MD5哈希对象

    chunk_size = 4096          #4kb分片，避免大文件一次性读入内存导致溢出
 
    try:                
        with open(filepath,"rb") as f:       #必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)         # 逐片更新哈希

            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"计算文件{filepath}md5失败，{str(e)}")
        return None                     # 捕获任何错误，记录日志后返回 None


def listdir_with_allowed_type(path:str,allowed_types:tuple[str]):       #返回文件夹内的文件列表（允许的文件后缀） 返回文件夹中指定后缀的文件列表
    files = []
 
    if not os.path.isdir(path):         #检查路径是否为文件
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return tuple()
    
    for f in os.listdir(path):     #获取文件夹下所有文件名
        if f.endswith(allowed_types):     #判断文件名是否以允许的后缀结尾，匹配则拼接完整路径并加入列表
            files.append(os.path.join(path,f))   
    
    return tuple(files)        #列表转为元组返回


def pdf_loader(filepath:str,passwd=None) -> list[Document]:   #加载 PDF 文件，返回 Document 对象列表
    return PyPDFLoader(filepath,passwd).load()       #.load()：解析 PDF，每页生成一个 Document 对象

def txt_loader(filepath:str,passwd=None) ->list[Document]:
    return TextLoader(filepath, encoding='utf-8').load()
    
