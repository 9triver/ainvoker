import pandas as pd
import os

file_path = "服务清单.xlsx"
output_dir = "ServiceList"
os.makedirs(output_dir, exist_k=True)

excel = pd.ExcelFile(file_path)

nodes = []
edges = []

def make_id(prefix, name):
    """生成唯一ID"""
    if pd.isna(name) or str(name).strip() == "":
        return None
    return f"{prefix}_{abs(hash(str(name))) % (10**8)}"

for sheet_name in excel.sheet_names:
    print(f"🔍 处理 sheet: {sheet_name}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    for col in ["接口一级分类", "接口开发单位", "开发负责人", "服务名称", "接口名称", "参数名", "参数类型"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    df = df.dropna(subset=["接口名称", "参数名"], how="any")

    for _, row in df.iterrows():
        cat_id = make_id("CAT", row.get("接口一级分类"))
        org_id = make_id("ORG", row.get("接口开发单位", sheet_name))
        person_id = make_id("PER", row.get("开发负责人"))
        service_id = make_id("SRV", row.get("服务名称"))
        api_id = make_id("API", row.get("接口名称"))
        param_key = f"{row.get('参数名')}_{row.get('参数中文名')}"
        param_id = make_id("PAR", param_key)

        contact_raw = row.get("联系方式")
        if pd.notna(contact_raw):
            contact = str(contact_raw).split(".")[0] if str(contact_raw).replace(".", "", 1).isdigit() else str(contact_raw)
        else:
            contact = ""

        nodes += [
            [cat_id, "InterfaceCategory", row.get("接口一级分类")],
            [org_id, "Organization", row.get("接口开发单位", sheet_name)],
            [person_id, "Person", row.get("开发负责人"), contact],
            [service_id, "Service", row.get("服务名称"), row.get("服务描述"), row.get("标准化服务名称"), row.get("标准化服务描述")],
            [api_id, "Interface", row.get("接口名称"), row.get("接口描述"), row.get("接口编码"),
             row.get("标准化接口名称"), row.get("标准化接口描述"), row.get("接口生产地址")]
        ]

        nodes += [[
            param_id, "Parameter",
            row.get("参数名"),
            row.get("参数中文名"),
            row.get("参数字段类型"),
            row.get("是否必填\n（true/false）"),
            row.get("格式"),
            row.get("是否枚举值"),
            row.get("码值对应")
        ]]

        edges += [
            [cat_id, api_id, "HAS_INTERFACE"],             # 分类 → 接口
            [org_id, person_id, "HAS_RESPONSIBLE"],        # 组织 → 人员
            [org_id, service_id, "PROVIDES_SERVICE"],      # 组织 → 服务
            [person_id, service_id, "RESPONSIBLE_FOR"],    # 人员 → 服务
            [service_id, api_id, "HAS_INTERFACE"]          # 服务 → 接口
        ]

        param_type_raw = str(row.get("参数类型", "")).lower()
        if any(k in param_type_raw for k in ["请求参数", "in", "request"]):
            edges.append([param_id, api_id, "INPUT_TO_INTERFACE"])
        elif any(k in param_type_raw for k in ["返回参数", "out", "response"]):
            edges.append([api_id, param_id, "OUTPUT_FROM_INTERFACE"])
        else:
            edges.append([api_id, param_id, "HAS_PARAMETER"])

nodes_df = pd.DataFrame(nodes, columns=["id", "label", "name", "attr1", "attr2", "attr3", "attr4", "attr5", "attr6"])
edges_df = pd.DataFrame(edges, columns=["start_id", "end_id", "type"])

nodes_df.drop_duplicates(subset=["id"], inplace=True)
edges_df.drop_duplicates(inplace=True)

nodes_path = os.path.join(output_dir, "nodes.csv")
edges_path = os.path.join(output_dir, "edges.csv")

nodes_df.to_csv(nodes_path, index=False, encoding="utf-8-sig")
edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")

print(f"✅ 已生成 {nodes_path} 和 {edges_path}，可直接导入 Neo4j。")



