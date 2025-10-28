from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Dict
import shutil
#from src.agents.agent_image2text import agent_flow
from src.agents.create_sales_order import agent_create_sales_order
import traceback, json, os

app = FastAPI()

@app.post("/query/image")
async def ask_agent_from_image(
    image: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        # Save image locally
        temp_path = Path("/tmp") / image.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)


        # Build the prompt
        input_prompt = f"{str(temp_path)}   \n{question}"
        agent_image2text = agent_create_sales_order()
        response = await agent_image2text.run_async(input_prompt, max_steps=5)

        final_answer = response.data["message"]["content"]["text"]
        print(final_answer)

        return JSONResponse(content={"final_answer": final_answer})
    
        # image_path = f"{PROJECT_ROOT}/images/orderhub_handwritten.jpg"
        # question = """
        # Get all information about the order.
        # """
        # # Read Order
        # input_prompt = image_path + "   " + question
        # response = agent_order.run(input_prompt, max_steps=4)
        # final_message = response.data["message"]["content"]["text"]
        # print(final_message)

    except Exception as e:
        # Print the full stack trace to stdout/logs
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )



@app.post("/orders/create")
async def create_sales_order(payload: Dict = Body(...)):
    """
    Create an order using the OCI AI agent and a structured JSON payload.
    """
    try:
        # Convert the Python dict to a properly escaped JSON string
        payload_json = json.dumps(payload)

        # Construct a human-readable prompt with embedded JSON
        input_prompt = f"Create a sales order using a properly structured JSON payload:\n{payload_json}"

        agent_order = agent_create_sales_order()
        response = await agent_order.run_async(input_prompt)

        final_answer = response.data["message"]["content"]["text"]
        print(final_answer)

        return JSONResponse(content={"final_answer": final_answer})
    except Exception as e:
        # Print the full stack trace to stdout/logs
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

@app.get("/orders/query")
async def query_sales_order(input_prompt: str):
    """
    Get sales order using a query string for the Oracle SCM API.
    Example:
    /orders/query?finder=findBySourceOrderNumberAndSystem;SourceTransactionNumber=404087,SourceTransactionSystem=GPR
    """
    try:
        agent_get = agent_create_sales_order()
        response = await agent_get.run_async(input_prompt)

        final_answer = response.data["message"]["content"]["text"]
        print(final_answer)

        return JSONResponse(content={"final_answer": final_answer})
    except Exception as e:
        # Print the full stack trace to stdout/logs
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )

from pydantic import BaseModel, Field

class SalesEmailRequest(BaseModel):
    saas_transaction_id: str | int = Field(..., description="Sales order id")
    final_message: str = Field(..., description="Email body content")

class SalesEmailResponse(BaseModel):
    email_string: str
    final_message_email: str

@app.post("/orders/email")
async def email_sales_order(payload: SalesEmailRequest):
    """
    Email the status of the Sales Order to a CSR
    """
    try:
        agent_order_email = agent_create_sales_order()
        

        saas_transaction_id = payload.saas_transaction_id
        final_message = payload.final_message

        input_prompt = (
            f"Send an email to ops@example.com: "
            f"subject: Sales Order Status for orderid : {saas_transaction_id}, "
            f"body: {final_message}"
        )

        #input_prompt = f"Send an email to ops@example.com: subject: Sales Order Created for orderid : {saas_transaction_id}, body: {final_message}"
    
        response = await agent_order_email.run_async(input_prompt, max_steps=3)

        final_answer = response.data["message"]["content"]["text"]
        print(final_answer)

        return JSONResponse(content={"final_answer": final_answer})
        
    except Exception as e:
        # Print the full stack trace to stdout/logs
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )