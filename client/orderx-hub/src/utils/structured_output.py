
import json
import re
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from src.llm.oci_genai import initialize_llm

llm = initialize_llm()

# Custom parser
def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between \`\`\`json and \`\`\` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"\`\`\`json(.*?)\`\`\`"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")


class BillToCustomer(BaseModel):
    """
    BillToCustomer
    """
    CustomerAccountId: Optional[int] = Field(None, description="Unique identifier for the customer account")
    SiteUseId: Optional[int] = Field(None, description="Site use identifier for billing")


class ShipToCustomer(BaseModel):
    """
    ShipToCustomer
    """
    PartyId: Optional[int] = Field(None, description="Party identifier for the shipping customer")
    SiteId: Optional[int] = Field(None, description="Site identifier for the shipping address")


class LineItem(BaseModel):
    """
    LineItem
    """
    SourceTransactionLineId: Optional[str] = Field(None, description="Unique identifier for the source transaction line")
    SourceTransactionLineNumber: Optional[str] = Field(None, description="Line number of the transaction")
    SourceScheduleNumber: Optional[str] = Field(None, description="Schedule number for the line")
    SourceTransactionScheduleId: Optional[str] = Field(None, description="Transaction schedule identifier")
    OrderedUOMCode: Optional[str] = Field(None, description="Unit of measure code for ordered quantity")
    OrderedQuantity: Optional[int] = Field(None, description="Quantity ordered")
    ProductNumber: Optional[str] = Field(None, description="Product number associated with the item")
    FOBPoint: Optional[str] = Field(None, description="Free on Board point")
    FreightTerms: Optional[str] = Field(None, description="Freight terms applicable to the line item")
    PaymentTerms: Optional[str] = Field(None, description="Payment terms for the transaction")
    ShipmentPriority: Optional[str] = Field(None, description="Shipment priority level")
    RequestedFulfillmentOrganizationId: Optional[int] = Field(None, description="ID of the requested fulfillment organization")
    ParentSourceTransactionLineId: Optional[str] = Field(None, description="Parent transaction line ID, if this line is dependent")
    PurchasingUOMCode: Optional[str] = Field(None, description="Purchasing unit of measure code")
    FOBPointCode: Optional[str] = Field(None, description="FOB point code for logistics")


class Transaction(BaseModel):
    """
    Transaction
    """
    SourceTransactionNumber: Optional[str] = Field(None, description="Transaction number from the source system")
    SourceTransactionSystem: Optional[str] = Field(None, description="Source system of the transaction")
    SourceTransactionId: Optional[str] = Field(None, description="Unique ID of the source transaction")
    TransactionalCurrencyCode: Optional[str] = Field(None, description="Currency code used for the transaction")
    BusinessUnitId: Optional[int] = Field(None, description="ID of the business unit initiating the transaction")
    BuyingPartyNumber: Optional[str] = Field(None, description="Number identifying the buying party")
    TransactionTypeCode: Optional[str] = Field(None, description="Code defining the type of transaction")
    RequestedShipDate: Optional[str] = Field(None, description="Date on which shipment is requested")
    SubmittedFlag: Optional[str] = Field(None, description="Flag indicating if the transaction is submitted")
    FreezePriceFlag: Optional[str] = Field(None, description="Flag to indicate if the price is frozen")
    FreezeShippingChargeFlag: Optional[str] = Field(None, description="Flag to indicate if the shipping charge is frozen")
    FreezeTaxFlag: Optional[str] = Field(None, description="Flag to indicate if tax is frozen")
    RequestingBusinessUnitId: Optional[int] = Field(None, description="ID of the business unit making the request")
    billToCustomer: Optional[List[BillToCustomer]] = Field(None, description="List of bill-to customer details")
    shipToCustomer: Optional[List[ShipToCustomer]] = Field(None, description="List of ship-to customer details")
    lines: Optional[List[LineItem]] = Field(None, description="List of transaction line items")


def llm_structured_output():
    trans_schema_str = json.dumps(Transaction.model_json_schema(), indent=2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the schema below to extract structured JSON from the user's request.\n"
                "Respond with valid JSON inside triple backticks like ```json ... ```.\n\n"
                "Schema:\n```json\n{trans_schema}\n```"
            ),
            ("human", "{query}")
        ]
    ).partial(trans_schema=trans_schema_str)

    query = """
BillToCustomer : 111000 
ShipToCustomer: 10124 Louisville
- Item: STOVE-ATO-Model, Quantity: 1, Requested Date: 20-JUL-25
- Item: GAS-FUEL, Quantity: 1, Requested Date: 20-JUL-25
- Item: BURNER-4-GRID, Quantity: 1, Requested Date: 20-JUL-25
"""

    chain = prompt | llm | extract_json

    try:
        response = chain.invoke({"query": query})
        print("🔍 Raw LLM Output:")
        print(json.dumps(response, indent=2))

        validated = Transaction.parse_obj(response[0])
        print("\n✅ Validated Structured Output:")
        print(validated.json(indent=2))
        return validated

    except ValidationError as e:
        print("❌ Schema validation failed:\n", e)
    except Exception as ex:
        print("❌ Failed to extract structured response:\n", ex)

    return response

if __name__ == "__main__":
    response = llm_structured_output()
    print(json.dumps(response, indent=2))