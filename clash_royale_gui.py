# clash_royale_gui.py
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QComboBox, QListWidget, 
                             QListWidgetItem, QTabWidget, QGroupBox, QProgressBar,
                             QMessageBox, QSplitter, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# Import your existing classifier code
from clash_royale_archetype_classifier import (
    QuickClashPredictor, find_card_id_by_name, 
    CLASH_ROYALE_CARDS, calculate_deck_stats
)

class PredictionThread(QThread):
    """Thread for running predictions to avoid GUI freezing"""
    prediction_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, predictor, input_type, input_data):
        super().__init__()
        self.predictor = predictor
        self.input_type = input_type
        self.input_data = input_data
    
    def run(self):
        try:
            if self.input_type == "url":
                result = self.predictor.predict_from_url(self.input_data)
            elif self.input_type == "deck_string":
                result = self.predictor.predict_from_deck_string(self.input_data)
            elif self.input_type == "card_names":
                result = self.predictor.predict_from_card_names(self.input_data)
            elif self.input_type == "card_ids":
                result = self.predictor.predict_from_card_ids(self.input_data)
            else:
                self.error_occurred.emit("Unknown input type")
                return
            
            self.prediction_finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

class CardWidget(QWidget):
    """Widget to display individual card information"""
    
    def __init__(self, card_id, card_info):
        super().__init__()
        self.card_id = card_id
        self.card_info = card_info
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Card name
        name_label = QLabel(self.card_info["name"])
        name_label.setFont(QFont("Arial", 10, QFont.Bold))
        name_label.setAlignment(Qt.AlignCenter)
        
        # Elixir cost
        elixir_label = QLabel(f"Elixir: {self.card_info['elixir']}")
        elixir_label.setAlignment(Qt.AlignCenter)
        
        # Type and rarity
        rarity_names = {1: "Common", 2: "Rare", 3: "Epic", 4: "Legendary", 5: "Champion"}
        type_label = QLabel(f"{self.card_info['type'].title()} - {rarity_names[self.card_info['rarity']]}")
        type_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(name_label)
        layout.addWidget(elixir_label)
        layout.addWidget(type_label)
        
        self.setLayout(layout)
        self.setStyleSheet("""
            CardWidget {
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 5px;
                margin: 2px;
                background-color: #2b2b2b;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
        """)

class DeckAnalysisWidget(QWidget):
    """Widget to display deck analysis results"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Archetype prediction
        self.archetype_label = QLabel("Archetype: Unknown")
        self.archetype_label.setFont(QFont("Arial", 14, QFont.Bold))
        
        self.confidence_label = QLabel("Confidence: 0%")
        self.confidence_label.setFont(QFont("Arial", 12))
        
        # Deck statistics
        stats_group = QGroupBox("Deck Statistics")
        stats_layout = QVBoxLayout()
        
        self.avg_elixir_label = QLabel("Average Elixir: -")
        self.four_card_cycle_label = QLabel("4-Card Cycle: -")
        self.total_elixir_label = QLabel("Total Elixir: -")
        
        stats_layout.addWidget(self.avg_elixir_label)
        stats_layout.addWidget(self.four_card_cycle_label)
        stats_layout.addWidget(self.total_elixir_label)
        stats_group.setLayout(stats_layout)
        
        # Card type distribution
        types_group = QGroupBox("Card Types")
        types_layout = QVBoxLayout()
        self.types_label = QLabel("Troops: 0, Spells: 0, Buildings: 0")
        types_layout.addWidget(self.types_label)
        types_group.setLayout(types_layout)
        
        # All probabilities
        self.probabilities_label = QLabel("")
        self.probabilities_label.setWordWrap(True)
        
        layout.addWidget(self.archetype_label)
        layout.addWidget(self.confidence_label)
        layout.addWidget(stats_group)
        layout.addWidget(types_group)
        layout.addWidget(self.probabilities_label)
        
        self.setLayout(layout)
    
    def update_analysis(self, prediction_result, deck_stats):
        """Update the analysis display with new results"""
        # Archetype and confidence
        self.archetype_label.setText(f"Archetype: {prediction_result['archetype'].title()}")
        self.confidence_label.setText(f"Confidence: {prediction_result['confidence']:.2%}")
        
        # Deck statistics
        self.avg_elixir_label.setText(f"Average Elixir: {deck_stats['average_elixir']:.2f}")
        self.four_card_cycle_label.setText(f"4-Card Cycle: {deck_stats['four_card_cycle']}")
        self.total_elixir_label.setText(f"Total Elixir: {deck_stats['total_elixir']}")
        
        # Card types
        card_types = prediction_result['card_types']
        self.types_label.setText(f"Troops: {card_types['troops']}, Spells: {card_types['spells']}, Buildings: {card_types['buildings']}")
        
        # Probabilities
        prob_text = "All Probabilities:\n"
        for arch, prob in sorted(prediction_result['all_probabilities'].items(), 
                               key=lambda x: x[1], reverse=True):
            prob_text += f"  {arch.title()}: {prob:.2%}\n"
        
        self.probabilities_label.setText(prob_text)

class ClashRoyaleGUI(QMainWindow):
    """Main GUI window for Clash Royale Archetype Classifier"""
    
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.current_deck = []
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        self.setWindowTitle("Clash Royale Archetype Classifier")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark theme
        self.set_dark_theme()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Input methods
        left_panel = self.create_input_panel()
        
        # Right panel - Results
        right_panel = self.create_results_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)
    
    def create_input_panel(self):
        """Create the input methods panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Clash Royale Deck Analyzer")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        # Tab widget for different input methods
        tabs = QTabWidget()
        
        # Tab 1: Manual Input
        manual_tab = self.create_manual_tab()
        
        # Tab 2: URL Input
        url_tab = self.create_url_tab()
        
        # Tab 3: Deck String
        deck_string_tab = self.create_deck_string_tab()
        
        tabs.addTab(manual_tab, "Manual Input")
        tabs.addTab(url_tab, "URL Input")
        tabs.addTab(deck_string_tab, "Deck String")
        
        # Current deck display
        self.deck_list = QListWidget()
        self.deck_list.setMaximumHeight(150)
        
        # Analyze button
        self.analyze_btn = QPushButton("Analyze Deck")
        self.analyze_btn.clicked.connect(self.analyze_deck)
        self.analyze_btn.setEnabled(False)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        layout.addWidget(title)
        layout.addWidget(tabs)
        layout.addWidget(QLabel("Current Deck:"))
        layout.addWidget(self.deck_list)
        layout.addWidget(self.analyze_btn)
        layout.addWidget(self.progress_bar)
        
        panel.setLayout(layout)
        return panel
    
    def create_manual_tab(self):
        """Create manual card input tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Card search
        search_layout = QHBoxLayout()
        self.card_search = QComboBox()
        self.card_search.setEditable(True)
        self.card_search.setInsertPolicy(QComboBox.NoInsert)
        
        # Populate with card names
        card_names = sorted([card_info["name"] for card_info in CLASH_ROYALE_CARDS.values()])
        self.card_search.addItems(card_names)
        
        self.add_card_btn = QPushButton("Add Card")
        self.add_card_btn.clicked.connect(self.add_manual_card)
        
        search_layout.addWidget(self.card_search)
        search_layout.addWidget(self.add_card_btn)
        
        # Clear deck button
        self.clear_deck_btn = QPushButton("Clear Deck")
        self.clear_deck_btn.clicked.connect(self.clear_deck)
        
        layout.addWidget(QLabel("Search and add cards:"))
        layout.addLayout(search_layout)
        layout.addWidget(self.clear_deck_btn)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_url_tab(self):
        """Create URL input tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.url_input = QTextEdit()
        self.url_input.setMaximumHeight(80)
        self.url_input.setPlaceholderText("Paste Clash Royale deck URL here...")
        
        self.load_url_btn = QPushButton("Load Deck from URL")
        self.load_url_btn.clicked.connect(self.load_from_url)
        
        layout.addWidget(QLabel("Deck URL:"))
        layout.addWidget(self.url_input)
        layout.addWidget(self.load_url_btn)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_deck_string_tab(self):
        """Create deck string input tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.deck_string_input = QTextEdit()
        self.deck_string_input.setMaximumHeight(80)
        self.deck_string_input.setPlaceholderText("Paste deck string (semicolon-separated card IDs)...")
        
        self.load_string_btn = QPushButton("Load Deck from String")
        self.load_string_btn.clicked.connect(self.load_from_string)
        
        layout.addWidget(QLabel("Deck String:"))
        layout.addWidget(self.deck_string_input)
        layout.addWidget(self.load_string_btn)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_results_panel(self):
        """Create the results display panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Analysis results
        self.analysis_widget = DeckAnalysisWidget()
        
        # Cards display area
        cards_label = QLabel("Deck Cards:")
        cards_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        # Scroll area for cards
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.cards_layout = QHBoxLayout()
        scroll_widget.setLayout(self.cards_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(120)
        
        layout.addWidget(self.analysis_widget)
        layout.addWidget(cards_label)
        layout.addWidget(scroll_area)
        
        panel.setLayout(layout)
        return panel
    
    def set_dark_theme(self):
        """Apply dark theme to the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: white;
            }
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #404040;
                color: white;
                padding: 8px;
                margin: 1px;
            }
            QTabBar::tab:selected {
                background-color: #555;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
            QComboBox, QTextEdit, QListWidget {
                background-color: #404040;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #4CAF50;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.predictor = QuickClashPredictor("clash_royale_classifier.pth")
            QMessageBox.information(self, "Success", "Model loaded successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Warning", 
                              f"Could not load model: {str(e)}\nPlease train a model first.")
    
    def add_manual_card(self):
        """Add card from manual search to deck"""
        if len(self.current_deck) >= 8:
            QMessageBox.warning(self, "Deck Full", "Deck already has 8 cards!")
            return
        
        card_name = self.card_search.currentText()
        card_id = find_card_id_by_name(card_name)
        
        if card_id:
            card_info = CLASH_ROYALE_CARDS[card_id]
            self.current_deck.append(card_id)
            self.update_deck_display()
        else:
            QMessageBox.warning(self, "Card Not Found", f"Could not find card: {card_name}")
    
    def clear_deck(self):
        """Clear the current deck"""
        self.current_deck.clear()
        self.update_deck_display()
    
    def update_deck_display(self):
        """Update the deck list and card widgets"""
        # Update list widget
        self.deck_list.clear()
        for card_id in self.current_deck:
            card_info = CLASH_ROYALE_CARDS[card_id]
            item = QListWidgetItem(f"{card_info['name']} ({card_info['elixir']} elixir)")
            self.deck_list.addItem(item)
        
        # Update card widgets
        self.update_card_widgets()
        
        # Enable/disable analyze button
        self.analyze_btn.setEnabled(len(self.current_deck) == 8)
    
    def update_card_widgets(self):
        """Update the card display widgets"""
        # Clear existing cards
        for i in reversed(range(self.cards_layout.count())): 
            self.cards_layout.itemAt(i).widget().setParent(None)
        
        # Add new cards
        for card_id in self.current_deck:
            card_info = CLASH_ROYALE_CARDS[card_id]
            card_widget = CardWidget(card_id, card_info)
            self.cards_layout.addWidget(card_widget)
    
    def load_from_url(self):
        """Load deck from URL"""
        url = self.url_input.toPlainText().strip()
        if not url:
            QMessageBox.warning(self, "Input Error", "Please enter a URL")
            return
        
        self.progress_bar.setVisible(True)
        self.analyze_btn.setEnabled(False)
        
        # Run prediction in thread
        self.prediction_thread = PredictionThread(self.predictor, "url", url)
        self.prediction_thread.prediction_finished.connect(self.handle_url_prediction)
        self.prediction_thread.error_occurred.connect(self.handle_prediction_error)
        self.prediction_thread.start()
    
    def handle_url_prediction(self, result):
        """Handle prediction result from URL"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        if 'error' in result:
            QMessageBox.warning(self, "Error", result['error'])
            return
        
        # Extract deck from result and update display
        # Note: This would need modification to your predictor to return the actual deck
        # For now, we'll just show the analysis results
        QMessageBox.information(self, "Analysis Complete", 
                              f"Deck classified as: {result['archetype'].title()}\n"
                              f"Confidence: {result['confidence']:.2%}")
    
    def load_from_string(self):
        """Load deck from string"""
        deck_string = self.deck_string_input.toPlainText().strip()
        if not deck_string:
            QMessageBox.warning(self, "Input Error", "Please enter a deck string")
            return
        
        try:
            card_ids = [int(card_id.strip()) for card_id in deck_string.split(';')]
            if len(card_ids) != 8:
                QMessageBox.warning(self, "Input Error", "Deck must contain exactly 8 cards")
                return
            
            # Validate card IDs
            valid_cards = []
            for card_id in card_ids:
                if card_id in CLASH_ROYALE_CARDS:
                    valid_cards.append(card_id)
                else:
                    QMessageBox.warning(self, "Invalid Card", f"Unknown card ID: {card_id}")
                    return
            
            self.current_deck = valid_cards
            self.update_deck_display()
            QMessageBox.information(self, "Success", "Deck loaded successfully!")
            
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid deck string format")
    
    def analyze_deck(self):
        """Analyze the current deck"""
        if len(self.current_deck) != 8:
            QMessageBox.warning(self, "Deck Error", "Deck must have exactly 8 cards")
            return
        
        self.progress_bar.setVisible(True)
        self.analyze_btn.setEnabled(False)
        
        # Run prediction in thread
        self.prediction_thread = PredictionThread(self.predictor, "card_ids", self.current_deck)
        self.prediction_thread.prediction_finished.connect(self.handle_prediction_result)
        self.prediction_thread.error_occurred.connect(self.handle_prediction_error)
        self.prediction_thread.start()
    
    def handle_prediction_result(self, result):
        """Handle successful prediction result"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        if 'error' in result:
            QMessageBox.warning(self, "Prediction Error", result['error'])
            return
        
        # Calculate deck statistics
        deck_stats = calculate_deck_stats(self.current_deck)
        
        # Update analysis display
        self.analysis_widget.update_analysis(result, deck_stats)
        
        # Show success message
        QMessageBox.information(self, "Analysis Complete", 
                              f"Deck classified as: {result['archetype'].title()}\n"
                              f"Confidence: {result['confidence']:.2%}")
    
    def handle_prediction_error(self, error_message):
        """Handle prediction errors"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Prediction Error", f"Error during prediction: {error_message}")

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = ClashRoyaleGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()