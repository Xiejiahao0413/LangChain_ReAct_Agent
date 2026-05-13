"""
为整个工程提供一个绝对路径
"""

import os 

def get_project_root() -> str:
    """
    获取工程所在的根目录
    ：return 字符串根目录
    """
    #获取当前文件绝对目录
    current_file = os.path.abspath(__file__)
    #获取工程的根目录，并获取文件所在的绝对路径
    current_dir = os.path.dirname(current_file)
    #获取工程根目录
    project_root = os.path.dirname(current_dir)

    return project_root


def get_abs_path(relative_path) -> str:
    """
    传入相对路径，得到绝对路径
    ：param relative_path:相对路径
    ：return: 绝对路径
    """ 
    project_root = get_project_root()
    return os.path.join(project_root,relative_path)

if __name__ == '__main__':
    print(get_abs_path("config/config.txt"))    #d:\pythonProject\LangChain_React_Agent\config/config.txt




