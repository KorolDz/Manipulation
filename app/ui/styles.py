APP_STYLESHEET = """
QMainWindow {
    background: #eef2f6;
    color: #111827;
    font-family: "Segoe UI", "Arial";
    font-size: 15px;
}

#rootFrame {
    background: #eef2f6;
}

#sidebar {
    background: #111827;
    border-radius: 8px;
}

#appTitle {
    color: #ffffff;
    font-size: 26px;
    font-weight: 800;
}

#pageTitle {
    color: #111827;
    font-size: 26px;
    font-weight: 800;
}

#panel,
#resultPanel {
    background: #ffffff;
    border: 1px solid #d9e1ea;
    border-radius: 8px;
}

#sectionTitle {
    color: #111827;
    font-size: 16px;
    font-weight: 800;
}

#fileNameLabel {
    color: #4b5563;
    font-size: 14px;
}

QPushButton {
    min-height: 40px;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 700;
    padding: 0 14px;
}

QPushButton:disabled {
    background: #e5e7eb;
    color: #9ca3af;
    border: 1px solid #d1d5db;
}

#primaryButton {
    background: #2563eb;
    color: #ffffff;
    border: 1px solid #2563eb;
}

#primaryButton:hover {
    background: #1d4ed8;
}

#secondaryButton {
    background: #ffffff;
    color: #2563eb;
    border: 1px solid #9db7f5;
}

#secondaryButton:hover {
    background: #eff6ff;
}

#clearButton {
    background: #f9fafb;
    color: #374151;
    border: 1px solid #d1d5db;
    min-height: 34px;
}

#clearButton:hover {
    background: #f3f4f6;
}

#exportButton {
    background: #111827;
    color: #ffffff;
    border: 1px solid #111827;
    min-height: 34px;
}

#exportButton:hover {
    background: #1f2937;
}

#statusReady,
#statusBusy,
#statusError {
    border-radius: 8px;
    font-size: 17px;
    font-weight: 800;
    padding: 14px;
}

#statusReady {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}

#statusBusy {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
}

#statusError {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}

#resultNeutral,
#resultSuccess,
#resultWarning,
#resultDanger {
    border-radius: 8px;
    font-size: 26px;
    font-weight: 900;
    padding: 24px;
}

#resultNeutral {
    background: #f8fafc;
    color: #334155;
    border: 1px solid #d9e1ea;
}

#resultSuccess {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}

#resultWarning {
    background: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
}

#resultDanger {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}

#resultMeta {
    color: #374151;
    font-size: 15px;
    font-weight: 600;
}

QTableWidget {
    background: #ffffff;
    color: #111827;
    border: 1px solid #d9e1ea;
    border-radius: 8px;
    gridline-color: #edf1f5;
    selection-background-color: #dbeafe;
    selection-color: #111827;
    font-size: 15px;
}

QHeaderView::section {
    background: #f8fafc;
    color: #475569;
    border: none;
    border-bottom: 1px solid #d9e1ea;
    padding: 8px;
    font-weight: 800;
    font-size: 14px;
}

QTableWidget::item {
    padding: 9px;
}
"""
