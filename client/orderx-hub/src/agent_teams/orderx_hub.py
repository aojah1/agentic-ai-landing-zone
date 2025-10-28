"""
orderx_hub.py
Author: Anup Ojah
Date: 2025-07-30
==========================
==Order Taking Supervisor Assistant==
==========================
This module is an order-taking supervisor assistant that can be used by a customers
to submit their sales orders through phone, email, handwritten notes or chat messages.
The agent will apply it's multi-modal tool capability, extract the required informaiton
to create an order in Fusion SCM

Workflow Overview:
1. Load config and credentials from .env
2. Register two sub agents to the supervisor - receive_sales_order and create_sales_order
3. Extract structured output from source, create an order, and send an email with the confirmed order to the sales agent
4. Run the agent with user input and print response
"""
