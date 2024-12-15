from typing import Union
import pathlib
import rich
import json
import os
import shutil

def load_json_file(file_path:Union[str , pathlib.PosixPath], data_type="dict"):
    if isinstance(file_path,pathlib.PosixPath):
        file_path = str(file_path)
    if data_type == "dict":
        json_file = dict()
    elif data_type == "list":
        json_file = list()
    else:
        raise ValueError("数据类型不对")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                json_file = json.load(f)
        except IOError as e:
            rich.print(f"[red]无法打开文件{file_path}：{e}")
        except json.JSONDecodeError as e:
            rich.print(f"[red]解析 JSON 文件时出错{file_path}：{e}")
    else:
        rich.print(f"[yellow]{file_path}文件不存在，正在传入空文件...[/yellow]")
    return json_file

def dump_json_file(json_file, file_path:Union[str , pathlib.PosixPath],indent=4,if_print = True,if_backup = True,if_backup_delete=False):
    if isinstance(file_path,pathlib.PosixPath):
        file_path = str(file_path)
    backup_path = file_path + ".bak"  # 定义备份文件的路径
    if os.path.exists(file_path) and if_backup:
        shutil.copy(file_path, backup_path)
    before_exist = os.path.exists(file_path)
    try:
        with open(file_path, 'w') as f:
            json.dump(json_file, f, indent=indent)
        if before_exist and if_print:
            rich.print(f"[yellow]更新{file_path}[/yellow]")
        elif if_print:
            rich.print(f"[green]创建{file_path}[/green]")
    except IOError as e:
        if os.path.exists(backup_path) and if_backup:
            shutil.copy(backup_path, file_path)
            if if_print:
                rich.print(f"[red]文件{file_path}写入失败，已从备份恢复原文件: {e}[/red]")
        else:
            if if_print:
                rich.print(f"[red]文件{file_path}写入失败，且无备份可用：{e}[/red]")
    finally:
        # 清理，删除备份文件
        if if_backup:
            if os.path.exists(backup_path) and if_backup_delete:
                os.remove(backup_path)
            if not os.path.exists(backup_path) and not if_backup_delete : #如果一开始是空的
                shutil.copy(file_path, backup_path)