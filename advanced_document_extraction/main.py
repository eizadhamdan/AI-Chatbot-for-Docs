import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
import fitz

load_dotenv()


class BoundingBoxField(BaseModel):
    bounding_box: list[int] = Field(
        ...,
        description="The bounding box where the information was found [y_min, x_min, y_max, x_max].",
    )
    page: int = Field(
        ..., description="The page number where the information was found."
    )


class TotalField(BoundingBoxField):
    total_value: float = Field(..., description="The total amount of the invoice.")


class RecipientField(BoundingBoxField):
    recipient_name: str = Field(..., description="The name of the recipient.")


class InvoiceModel(BaseModel):
    total: TotalField
    recipient: RecipientField


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

pdf = client.files.upload(file="documents/invoice.pdf")

prompt = """"
    Extract the invoice recipient name and invoice total.
    Return only JSON that matches the provided schema.
    If a field is missing, set it to null (and bounding box to [0, 0, 0, 0]).
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[pdf, prompt],
    config={"response_mime_type": "application/json", "response_schema": InvoiceModel},
)

invoice = InvoiceModel.model_validate_json(response.text)

print(invoice.model_dump())

items_to_draw = [
    ("TOTAL", invoice.total.bounding_box, invoice.total.page),
    ("RECIPIENT", invoice.recipient.bounding_box, invoice.recipient.page),
]

for document_name in ["documents/invoice.pdf", "documents/invoice_multipage.pdf"]:
    document = fitz.open(document_name)

    for label, box, page_number in items_to_draw:
        if not box or box == [0, 0, 0, 0]:
            continue

        page = document[page_number - 1]
        y0, x0, y1, x1 = box

        r = page.rect

        rect = fitz.Rect(
            (x0 / 1000) * r.width,
            (y0 / 1000) * r.height,
            (x1 / 1000) * r.width,
            (y1 / 1000) * r.height,
        )

        page.draw_rect(rect, color=(1, 0, 0), width=2)
        page.insert_text((rect.x0, rect.y0 - 2), label, color=(1, 0, 0), fontsize=6)

    document.save(f"{document_name[:-4]}_annotated.pdf")
    document.close()
