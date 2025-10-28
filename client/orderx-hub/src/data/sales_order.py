from pydantic import BaseModel, Field
from typing import List, Optional

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
