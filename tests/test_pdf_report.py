import unittest

from app.ui.main_window import PDF_EVIDENCE_IMAGE_WIDTH_PT, PDF_MARGIN_MM


class PdfReportLayoutTests(unittest.TestCase):
    def test_pdf_margin_is_15_mm(self):
        self.assertEqual(PDF_MARGIN_MM, 15)

    def test_pdf_evidence_image_fits_printable_width(self):
        self.assertLessEqual(PDF_EVIDENCE_IMAGE_WIDTH_PT, 460)


if __name__ == "__main__":
    unittest.main()
