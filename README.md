# 数据处理脚本调用顺序
## excel数据导入数据库
```bash
python data_process/excel_to_neo4j.py

python data_process/change_attr.py

python data_process/add_examples.py
```

## 大模型数据增强处理
```bash
python data_process/add_param_description.py

python data_process/add_interface_description.py

python data_process/add_interface_struct_description.py

```


# 环境变量
```bash
export EMBED_BASE_URL=http://xxx.xxx.xxx.xxx:1234/v1
```

# 节点嵌入
```bash
python embed/embed_service-list.py
```

# 问答
```bash
python agent.py
```