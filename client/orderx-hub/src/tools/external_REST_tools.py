# Connect to any service to integrate any data and functionality with a public REST interface
import requests, os
import json
from oci.addons.adk import Toolkit, tool
from pathlib import Path
from dotenv import load_dotenv
from src.common.config import *


@tool
def create_sales_order(payload: dict) -> str:
    """
    You are a tools to create sales order by invoking an External REST API.
    :param query:
    :return:
    """
    try:

        headers = {
            "Content-Type": "application/vnd.oracle.adf.resourceitem+json",
            "Accept": "application/vnd.oracle.adf.resourceitem+json"
        }

        if isinstance(payload, dict):
            payload = json.dumps(payload)

        response = requests.post(
            API_URL,
            auth=(API_USER, API_PASS),
            headers=headers,
            data=payload  # ✅ Correctly serialized JSON
        )

        print("Status Code:", response.status_code)
        print("Response Text:", response.text)

        response.raise_for_status()
        return f"Response: {json.dumps(response.json(), indent=4)}"

    except requests.exceptions.RequestException as e:
        return f"API call failed: {str(e)}"

@tool
def get_sales_order(orderid: str) -> str:
    """
    You are a tools to get sales order by invoking an External REST API.
    :param query:
    :return:
    """
    try:

        """
        curl -u username:password "https://servername/fscmRestApi/resources/version/salesOrdersForOrderHub"
        """
        response = requests.get(
            API_URL + "?" + orderid,
            auth=(API_USER, API_PASS),
        )

        print("Status Code:", response.status_code)
        print("Response Text:", response.text)

        response.raise_for_status()
        return f"Response: {json.dumps(response.json(), indent=4)}"

    except requests.exceptions.RequestException as e:
        return f"API call failed: {str(e)}"
    

@tool
def fetch_sales_order(order_key: str) -> str:
    """
    You are a tools to fetch one sales order using order_key by invoking an External REST API.
    :param query:
    :return:
    """
    try:

        """
        curl -u username:password "https://servername/fscmRestApi/resources/version/salesOrdersForOrderHub"
        """
        response = requests.get(
            API_URL + "?" + order_key,
            auth=(API_USER, API_PASS),
        )

        print("Status Code:", response.status_code)
        print("Response Text:", response.text)

        response.raise_for_status()
        return f"Response: {json.dumps(response.json(), indent=4)}"

    except requests.exceptions.RequestException as e:
        return f"API call failed: {str(e)}"

def test_case_create_sales_order():
    payload = {
        "SourceTransactionNumber": "R13_Sample_Order_ATOModel_01",
        "SourceTransactionSystem": "GPR",
        "SourceTransactionId": "R13_Sample_Order_ATOModel_01",
        "TransactionalCurrencyCode": "USD",
        "BusinessUnitId": 204,
        "BuyingPartyNumber": "1006",
        "TransactionTypeCode": "STD",
        "RequestedShipDate": "2018-09-19T19:51:48+00:00",
        "SubmittedFlag": 'true',
        "FreezePriceFlag": 'false',
        "FreezeShippingChargeFlag": 'false',
        "FreezeTaxFlag": 'false',
        "RequestingBusinessUnitId": 204,
        "billToCustomer": [{
            "CustomerAccountId": 1006,
            "SiteUseId": 1025
        }],
        "shipToCustomer": [{
            "PartyId": 1006,
            "SiteId": 1036
        }],
        "lines": [{
            "SourceTransactionLineId": "1",
            "SourceTransactionLineNumber": "1",
            "SourceScheduleNumber": "1",
            "SourceTransactionScheduleId": "1",
            "OrderedUOMCode": "Ea",
            "OrderedQuantity": 1,
            "ProductNumber": "STOVE_ATO_MODEL",
            "FOBPoint": "Destination",
            "FreightTerms": "Add freight",
            "PaymentTerms": "30 Net",
            "ShipmentPriority": "High",
            "RequestedFulfillmentOrganizationId": 204
        },
        {
            "SourceTransactionLineId": "2",
            "SourceTransactionLineNumber": "2",
            "SourceScheduleNumber": "1",
            "SourceTransactionScheduleId": "1",
            "OrderedUOMCode": "Ea",
            "OrderedQuantity": 1,
            "ProductNumber": "GAS_FUEL",
            "FOBPoint": "Destination",
            "FreightTerms": "Add freight",
            "PaymentTerms": "30 Net",
            "ShipmentPriority": "High",
            "RequestedFulfillmentOrganizationId": 204,
            "ParentSourceTransactionLineId": "1"
        },
        {
            "SourceTransactionLineId": "3",
            "SourceTransactionLineNumber": "3",
            "SourceScheduleNumber": "1",
            "SourceTransactionScheduleId": "1",
            "OrderedUOMCode": "Ea",
            "OrderedQuantity": 1,
            "ProductNumber": "Burner_4_GRID",
            "PurchasingUOMCode": "Ea",
            "FOBPoint": "Destination",
            "FreightTerms": "Add freight",
            "PaymentTerms": "30 Net",
            "ShipmentPriority": "High",
            "FOBPointCode":"Destination",
            "RequestedFulfillmentOrganizationId": 204,
            "ParentSourceTransactionLineId": "1"
        }
        ]
    }


    create_sales_order(payload)

def test_get_sales_order():
    order_string = "finder=findBySourceOrderNumberAndSystem;SourceTransactionNumber=404087,SourceTransactionSystem=GPR"
    get_sales_order(order_string)

def test_fetch_sales_order():
    order_string = "q=OrderNumber=97412"
    fetch_sales_order(order_string)

if __name__ == "__main__":
    test_fetch_sales_order()
    #test_get_sales_order()