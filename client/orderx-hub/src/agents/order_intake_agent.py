"""
order_intake_agent.py
Author: Anup Ojah
Date: 2025-10-26
==========================
==Order Taking Assistant==
==========================
This module is an order-taking assistant designed to support sales workflows by extracting
customer order information from uploaded images and creates a sales order by posting data to APIs such as Fusion SCM
Workflow Overview:
1. Load config and credentials from .env
2. Register tools with the agent - create_sales_order, get_sales_order, sales_order_email
3. Extract structured output from image_to_text to be able to create an order
4. Run the agent with user input and print response
5. Send confirmation email after an Order is created
"""
