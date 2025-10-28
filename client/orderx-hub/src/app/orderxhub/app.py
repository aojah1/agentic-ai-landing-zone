
import json, requests, streamlit as st

st.set_page_config(page_title="Agent Orchestrator (Streaming)", page_icon="🤖", layout="wide")
st.title("🤖 ERP - Sales Order Automation Agent")
st.caption("Master agent executes sub agents in serial with streaming logs and a LangGraph-style diagram. (No Python graphviz dependency required)")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🔧 Configuration")
    base_url = st.text_input("Base URL", value="http://0.0.0.0:8084", help="Root of your API (no trailing slash)")
    base_url = base_url.rstrip('/')
    timeout = st.number_input("HTTP Timeout (s)", value=60, min_value=1, max_value=600)
    st.markdown("---")
    st.subheader("Run server")
    st.code("streamlit run app.py --server.address 0.0.0.0 --server.port 8505")
    st.markdown("---")
    transaction_number = st.text_input("Transaction Number", value="R230_Sample_Order_ATOModel_232")

session = requests.Session()


def _url(path: str) -> str:
    return f"{base_url}{path if path.startswith('/') else '/' + path}"

def GET(path, **kw):
    return session.get(_url(path), timeout=timeout, **kw)

def POST(path, **kw):
    return session.post(_url(path), timeout=timeout, **kw)

# ---------------- Preset Tools ----------------
st.subheader("🛠️ Tools")
st.caption("Wired to your endpoints: /query/image, /orders/create, /orders/query")

# Tool inputs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**1) Text_2_Image — Agent **")
    q_question = st.text_input("question", value="Get all information about the order", key="q1")
    q_image = st.file_uploader("image", type=["png","jpg","jpeg"], key="f1")

with col2:
    st.markdown("**2) Create_Sales_Order — Agent **")
    default_body = {
    "SourceTransactionNumber": transaction_number,
    "SourceTransactionSystem": "OPS",
    "SourceTransactionId": transaction_number,
    "TransactionalCurrencyCode": "USD",
    "BusinessUnitId": 300000046987012,
    "BuyingPartyNumber": "10060",
    "RequestedShipDate": "2018-09-19",
    "SubmittedFlag": "true",
    "FreezePriceFlag": "false",
    "FreezeShippingChargeFlag": "false",
    "FreezeTaxFlag": "false",
    "RequestingBusinessUnitId": 300000046987012,
    "lines": [
        {
        "SourceTransactionLineId": "1",
        "SourceTransactionLineNumber": "1",
        "SourceScheduleNumber": "1",
        "SourceTransactionScheduleId": "1",
        "OrderedUOMCode": "zzu",
        "OrderedQuantity": 5,
        "ProductNumber": "AS6647431",
        "FOBPoint": "Destination",
        "FreightTerms": "Add freight",
        "PaymentTerms": "30 Net",
        "ShipmentPriority": "High"
        },
        {
        "SourceTransactionLineId": "2",
        "SourceTransactionLineNumber": "2",
        "SourceScheduleNumber": "1",
        "SourceTransactionScheduleId": "1",
        "OrderedUOMCode": "zzu",
        "OrderedQuantity": 10,
        "ProductNumber": "AS6647432",
        "FOBPoint": "Destination",
        "FreightTerms": "Add freight",
        "PaymentTerms": "30 Net",
        "ShipmentPriority": "High"
        },
        {
        "SourceTransactionLineId": "3",
        "SourceTransactionLineNumber": "3",
        "SourceScheduleNumber": "1",
        "SourceTransactionScheduleId": "1",
        "OrderedUOMCode": "zzu",
        "OrderedQuantity": 15,
        "ProductNumber": "AS6647433",
        "FOBPoint": "Destination",
        "FreightTerms": "Add freight",
        "PaymentTerms": "30 Net",
        "ShipmentPriority": "High"
        }
    ]
}
    q_body_raw = st.text_area("Order JSON", value=json.dumps(default_body, indent=2), height=260, key="body")

with col3:
    st.markdown("**3) Get_Sales_Order — Agent **")
    q_order_id = st.text_input("Order ID", value=transaction_number, key="oid")

with col4:
    st.markdown("**4) Sales_Order_Email — Agent **")
    q_order_id = st.text_input("Order ID", value=transaction_number, key="oid_email")


st.markdown("---")

# ---------------- Top Diagram (LangGraph-like) ----------------
diagram_placeholder = st.empty()

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAIL = "fail"

def render_graph(status_map):
    colors = {
        STATUS_PENDING: "#e5e7eb",
        STATUS_RUNNING: "#fde68a",
        STATUS_SUCCESS: "#bbf7d0",
        STATUS_FAIL: "#fecaca",
    }
    dot = []
    dot.append('digraph G {')
    dot.append('rankdir="LR";')
    dot.append('node [fontname="Helvetica"];')
    dot.append('A [label="Master Agent \n OrderX Hub", shape="circle", style="filled", fillcolor="#e0e7ff"];')
    def add_tool(code, label):
        stt = status_map.get(code, STATUS_PENDING)
        dot.append(f'{code} [label="{label}", shape="box", style="filled,rounded", fillcolor="{colors[stt]}"];')
    add_tool("T1", "Text_2_Image\n Agent")
    add_tool("T2", "Create_Sales_Order\n Agent")
    add_tool("T3", "Get_Sales_Order\n Agent")
    add_tool("T4", "Sales_Order_Email\n Agent")
    dot.append("A -> T1;")
    dot.append("T1 -> T2;")
    dot.append("T2 -> T3;")
    dot.append("T3 -> T4;")
    dot.append("}")
    dot_src = "\n".join(dot)
    diagram_placeholder.graphviz_chart(dot_src)

status_map = {"T1": STATUS_PENDING, "T2": STATUS_PENDING, "T3": STATUS_PENDING, "T4": STATUS_PENDING}
render_graph(status_map)

# ---------------- Streaming Logs + Execute ----------------
run_col, log_col = st.columns([1,2])
with run_col:
    run = st.button("🚀 Execute Master Agent - OrderX Hub")

with log_col:
    st.markdown("**Live Logs**")
    log_area = st.empty()

def stream_log(line=None, reset=False):
    if reset or "log_lines" not in st.session_state:
        st.session_state["log_lines"] = []
    if line is not None:
        st.session_state["log_lines"].append(line)
    content = "\n".join(st.session_state["log_lines"][-500:])
    log_area.code(content)

def show_payload(title, payload):
    st.markdown(f"**{title}**")
    if isinstance(payload, (dict, list)):
        st.json(payload)
    else:
        st.code(str(payload)[:5000])

if run:
    stream_log(reset=True)

    # Step 1
    status_map["T1"] = STATUS_RUNNING
    render_graph(status_map)
    stream_log("Step 1/4: Text_2_Image — uploading image + question…")
    try:
        files = None
        if q_image is not None:
            files = {"image": (q_image.name, q_image.getvalue(), q_image.type or "image/jpeg")}
        data = {"question": q_question}
        r1 = POST("/query/image", files=files, data=data, headers={"accept":"application/json"})
        try:
            p1 = r1.json() if r1.headers.get("content-type","").startswith("application/json") else r1.text
        except Exception:
            p1 = r1.text
        ok1 = r1.ok
        stream_log(f"Step 1 → HTTP {r1.status_code}")
        status_map["T1"] = STATUS_SUCCESS if ok1 else STATUS_FAIL
        render_graph(status_map)
        show_payload("/query/image response", p1)
        if not ok1:
            stream_log("Halting due to failure in Step 1.")
    except Exception as e:
        status_map["T1"] = STATUS_FAIL
        render_graph(status_map)
        stream_log(f"Step 1 error: {e}")

    if status_map["T1"] == STATUS_SUCCESS:
        # Step 2
        status_map["T2"] = STATUS_RUNNING
        render_graph(status_map)
        stream_log("Step 2/4: Create_Sales_Order — posting order JSON…")
        try:
            payload2 = json.loads(q_body_raw)
            r2 = POST("/orders/create", json=payload2, headers={"Content-Type":"application/json","accept":"application/json"})
            try:
                p2 = r2.json() if r2.headers.get("content-type","").startswith("application/json") else r2.text
            except Exception:
                p2 = r2.text
            ok2 = r2.ok
            stream_log(f"Step 2 → HTTP {r2.status_code}")
            status_map["T2"] = STATUS_SUCCESS if ok2 else STATUS_FAIL
            render_graph(status_map)
            show_payload("/orders/create response", p2)
        except Exception as e:
            status_map["T2"] = STATUS_FAIL
            render_graph(status_map)
            stream_log(f"Step 2 error: {e}")

    if status_map["T2"] == STATUS_SUCCESS:
        # Step 3
        status_map["T3"] = STATUS_RUNNING
        render_graph(status_map)
        derived_id = q_order_id
        try:
            if isinstance(p2, dict):
                derived_id = p2.get("OrderNumber") or p2.get("id") or p2.get("SourceTransactionNumber") or derived_id
        except Exception:
            pass
        stream_log(f"Step 3/4: Get_Sales_Order — querying order: {derived_id}")
        try:
            prompt = f"get sales order for order id : {derived_id}"
            r3 = GET("/orders/query", params={"input_prompt": prompt}, headers={"accept":"application/json"})
            try:
                p3 = r3.json() if r3.headers.get("content-type","").startswith("application/json") else r3.text
            except Exception:
                p3 = r3.text
            ok3 = r3.ok
            stream_log(f"Step 3 → HTTP {r3.status_code}")
            status_map["T3"] = STATUS_SUCCESS if ok3 else STATUS_FAIL
            render_graph(status_map)
            show_payload("/orders/query response", p3)
        except Exception as e:
            status_map["T3"] = STATUS_FAIL
            render_graph(status_map)
            stream_log(f"Step 3 error: {e}")
    
    if status_map["T3"] == STATUS_SUCCESS:
        # Step 2
        status_map["T4"] = STATUS_RUNNING
        render_graph(status_map)
        stream_log("Step 4/4: Sales_Order_Email — posting order JSON…")
        try:
            #payload4 = json.loads(p3)
            # Build the *exact* payload the FastAPI endpoint requires
            payload4 = {
                "saas_transaction_id": derived_id,     # int or str is fine
                "final_message": str(r3.text),            # ensure string
            }

            r4 = POST("/orders/email", json=payload4, headers={"Content-Type":"application/json","accept":"application/json"})
            try:
                p4 = r4.json() if r4.headers.get("content-type","").startswith("application/json") else r4.text
            except Exception:
                p4 = r4.text
            ok4 = r4.ok
            stream_log(f"Step 4 → HTTP {r4.status_code}")
            status_map["T4"] = STATUS_SUCCESS if ok4 else STATUS_FAIL
            render_graph(status_map)
            show_payload("/orders/email response", p4)
        except Exception as e:
            status_map["T4"] = STATUS_FAIL
            render_graph(status_map)
            stream_log(f"Step 4 error: {e}")

    stream_log("Done.")
