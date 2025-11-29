# clash_royale_archetype_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests
import json
from typing import List, Dict, Tuple, Set
from collections import Counter
import re
from urllib.parse import unquote
import pickle

#https://deckbandit.com/top-decks/pathOfLegend/*/season


class ClashRoyaleDataProcessor:
    def __init__(self):
        self.card_types = {
            'troop': 26,
            'building': 27,
            'spell': 28
        }

        # Archetype definitions
        self.archetypes = [
            'beatdown', 'control', 'siege', 'bridge_spam',
            'cycle', 'bait', 'split_lane'
        ]

        self.archetype_to_idx = {arch: i for i, arch in enumerate(self.archetypes)}
        self.idx_to_archetype = {i: arch for i, arch in enumerate(self.archetypes)}

        # We'll build this dynamically from training data
        self.all_card_ids: Set[int] = set()
        self.card_id_to_index: Dict[int, int] = {}
        self.index_to_card_id: Dict[int, int] = {}

    def extract_deck_from_url(self, url: str) -> List[int]:
        """Extract card IDs from Clash Royale deck URLs"""
        try:
            # Handle Clash Royale deep link format
            if 'clashroyale://copyDeck' in url:
                # Extract the deck parameter from the URL
                deck_match = re.search(r'deck=([^&]+)', url)
                if deck_match:
                    deck_string = deck_match.group(1)
                    # Split by semicolons and convert to integers
                    card_ids = [int(card_id) for card_id in deck_string.split(';')]
                    deck = card_ids[:8]  # Take first 8 cards
                else:
                    return []

            # For royaleapi format (API response)
            elif 'royaleapi.com' in url and '/deck/' in url:
                try:
                    # Add .json to get API response
                    if not url.endswith('.json'):
                        api_url = url + '.json'
                    else:
                        api_url = url

                    response = requests.get(api_url)
                    data = response.json()
                    deck = [card['id'] for card in data.get('cards', [])]
                except:
                    # Fallback: try to extract from HTML
                    response = requests.get(url)
                    html_content = response.text
                    card_ids = re.findall(r'"id":(\d{8})', html_content)
                    deck = [int(card_id) for card_id in card_ids[:8]]

            # For deckbandit format
            elif 'deckbandit' in url:
                response = requests.get(url)
                html_content = response.text
                card_ids = re.findall(r'"id":(\d{8})', html_content)
                deck = [int(card_id) for card_id in card_ids[:8]]

            else:
                # Generic fallback - look for 8-digit numbers
                response = requests.get(url)
                text_content = response.text
                card_ids = re.findall(r'\b(26\d{6}|27\d{6}|28\d{6})\b', text_content)
                deck = [int(card_id) for card_id in card_ids[:8]]

            # Validate we got exactly 8 cards
            if len(deck) != 8:
                print(f"Warning: Got {len(deck)} cards instead of 8 from {url}")
                return []

            # Add these card IDs to our master list
            for card_id in deck:
                self.all_card_ids.add(card_id)

            return deck

        except Exception as e:
            print(f"Error extracting deck from {url}: {e}")
            return []

    def extract_from_deck_string(self, deck_string: str) -> List[int]:
        """Extract card IDs from a deck string (semicolon-separated)"""
        try:
            card_ids = [int(card_id.strip()) for card_id in deck_string.split(';')]
            if len(card_ids) == 8:
                for card_id in card_ids:
                    self.all_card_ids.add(card_id)
                return card_ids
            else:
                print(f"Warning: Deck string has {len(card_ids)} cards, expected 8")
                return []
        except Exception as e:
            print(f"Error parsing deck string: {e}")
            return []

    def build_card_mapping(self):
        """Build mapping from actual card IDs to sequential indices"""
        self.card_id_to_index = {}
        self.index_to_card_id = {}

        for idx, card_id in enumerate(sorted(self.all_card_ids)):
            self.card_id_to_index[card_id] = idx
            self.index_to_card_id[idx] = card_id

        print(f"Built mapping for {len(self.card_id_to_index)} unique cards")

    def deck_to_vector(self, deck: List[int]) -> torch.Tensor:
        """Convert deck to one-hot encoded vector using actual card IDs"""
        if not self.card_id_to_index:
            self.build_card_mapping()

        vector = torch.zeros(len(self.card_id_to_index))

        for card_id in deck:
            if card_id in self.card_id_to_index:
                vector[self.card_id_to_index[card_id]] = 1
            else:
                print(f"Warning: Unknown card ID {card_id}")

        return vector

    def get_card_type_distribution(self, deck: List[int]) -> torch.Tensor:
        """Get distribution of card types in deck based on ID prefixes"""
        distribution = torch.zeros(3)  # [troops, spells, buildings]

        for card_id in deck:
            # Extract first two digits to determine type
            prefix = int(str(card_id)[:2])

            if prefix == 26:
                distribution[0] += 1  # troop
            elif prefix == 27:
                distribution[2] += 1  # building
            elif prefix == 28:
                distribution[1] += 1  # spell
            else:
                print(f"Warning: Unknown card prefix {prefix} for ID {card_id}")

        return distribution / 8.0  # Normalize

    def get_card_id_type(self, card_id: int) -> str:
        """Get card type from ID"""
        prefix = int(str(card_id)[:2])

        if prefix == 26:
            return "troop"
        elif prefix == 27:
            return "building"
        elif prefix == 28:
            return "spell"
        else:
            return "unknown"


class DynamicDeckClassifier(nn.Module):
    def __init__(self, card_vocab_size=121, type_input_size=3, hidden_size=256, num_classes=7):
        super(DynamicDeckClassifier, self).__init__()

        # Card composition branch - dynamic input size
        self.card_branch = nn.Sequential(
            nn.Linear(card_vocab_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )

        # Card type distribution branch
        self.type_branch = nn.Sequential(
            nn.Linear(type_input_size, hidden_size // 4),
            nn.ReLU()
        )

        # Combined features
        combined_size = hidden_size // 2 + hidden_size // 4

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, card_features, type_features):
        card_out = self.card_branch(card_features)
        type_out = self.type_branch(type_features)

        combined = torch.cat([card_out, type_out], dim=1)
        return self.classifier(combined)


class ArchetypeTrainer:
    def __init__(self):
        self.processor = ClashRoyaleDataProcessor()
        self.model = None
        self.criterion = nn.CrossEntropyLoss()

    def load_training_data(self, deck_data: List[Dict]):
        """Load training data from list of dictionaries containing URLs and archetypes"""
        self.deck_vectors = []
        self.type_vectors = []
        self.labels = []

        print("Extracting card IDs from training data...")
        successful_decks = 0

        for i, data in enumerate(deck_data):
            if i % 10 == 0:
                print(f"Processed {i}/{len(deck_data)} decks...")

            deck = None
            url = data.get('url', '')
            archetype = data.get('archetype', '')

            if not archetype:
                print(f"Warning: No archetype for deck {i}")
                continue

            if url:
                deck = self.processor.extract_deck_from_url(url)

            # If URL extraction failed but we have a deck string, use that
            if not deck and 'deck_string' in data:
                deck = self.processor.extract_from_deck_string(data['deck_string'])

            if deck and len(deck) == 8:
                # Store deck for later vectorization
                self.deck_vectors.append(deck)
                type_vector = self.processor.get_card_type_distribution(deck)
                label_idx = self.processor.archetype_to_idx[archetype]

                self.type_vectors.append(type_vector)
                self.labels.append(label_idx)
                successful_decks += 1
            else:
                print(f"Warning: Could not extract valid deck from item {i}")

        if successful_decks == 0:
            raise ValueError("No valid decks could be extracted from training data")

        # Build card mapping after collecting all IDs
        self.processor.build_card_mapping()

        # Now convert decks to vectors
        print("Converting decks to feature vectors...")
        deck_feature_vectors = []
        for deck in self.deck_vectors:
            deck_feature_vectors.append(self.processor.deck_to_vector(deck))

        # Convert to tensors
        self.X_cards = torch.stack(deck_feature_vectors)
        self.X_types = torch.stack(self.type_vectors)
        self.y = torch.tensor(self.labels)

        # Initialize model with correct input size
        vocab_size = len(self.processor.card_id_to_index)
        self.model = DynamicDeckClassifier(
            card_vocab_size=vocab_size,
            num_classes=len(self.processor.archetypes)
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        print(f"Successfully loaded {successful_decks} training examples")
        print(f"Vocabulary size: {vocab_size} unique cards")

    def train(self, epochs=100, validation_split=0.2):
        """Train the model"""
        if self.model is None:
            raise ValueError("Must load training data first")

        # Split data
        dataset_size = len(self.X_cards)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))

        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            self.X_cards[train_indices],
            self.X_types[train_indices],
            self.y[train_indices]
        )
        val_dataset = torch.utils.data.TensorDataset(
            self.X_cards[val_indices],
            self.X_types[val_indices],
            self.y[val_indices]
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

        # Training loop
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for batch_cards, batch_types, batch_labels in train_loader:
                self.optimizer.zero_grad()

                outputs = self.model(batch_cards, batch_types)
                loss = self.criterion(outputs, batch_labels)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_cards, batch_types, batch_labels in val_loader:
                    outputs = self.model(batch_cards, batch_types)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)
            train_losses.append(epoch_loss / len(train_loader))

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}, '
                      f'Val Accuracy: {accuracy:.2f}%')

        self.scheduler.step()

        return train_losses, val_accuracies

    def predict_deck(self, deck: List[int]) -> Dict:
        """Predict archetype for a deck"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        self.model.eval()

        with torch.no_grad():
            deck_vector = self.processor.deck_to_vector(deck).unsqueeze(0)
            type_vector = self.processor.get_card_type_distribution(deck).unsqueeze(0)

            outputs = self.model(deck_vector, type_vector)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            archetype = self.processor.idx_to_archetype[predicted.item()]

            # Get all probabilities
            all_probs = {}
            for i, arch in self.processor.idx_to_archetype.items():
                all_probs[arch] = probabilities[0][i].item()

            return {
                'archetype': archetype,
                'confidence': confidence.item(),
                'all_probabilities': all_probs,
                'card_types': {
                    'troops': sum(1 for card in deck if self.processor.get_card_id_type(card) == 'troop'),
                    'spells': sum(1 for card in deck if self.processor.get_card_id_type(card) == 'spell'),
                    'buildings': sum(1 for card in deck if self.processor.get_card_id_type(card) == 'building')
                }
            }

    def save_model(self, filepath: str):
        """Save trained model and processor state"""
        if self.model is None:
            raise ValueError("No model to save")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'processor_state': {
                'card_id_to_index': self.processor.card_id_to_index,
                'index_to_card_id': self.processor.index_to_card_id,
                'all_card_ids': list(self.processor.all_card_ids),
                'archetypes': self.processor.archetypes,
                'archetype_to_idx': self.processor.archetype_to_idx
            }
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model and processor state"""
        checkpoint = torch.load(filepath)

        # Restore processor state
        if 'processor_state' in checkpoint:
            state = checkpoint['processor_state']
            self.processor.card_id_to_index = state['card_id_to_index']
            self.processor.index_to_card_id = state['index_to_card_id']
            self.processor.all_card_ids = set(state['all_card_ids'])
            self.processor.archetypes = state['archetypes']
            self.processor.archetype_to_idx = state['archetype_to_idx']
            self.processor.idx_to_archetype = {v: k for k, v in self.processor.archetype_to_idx.items()}

        # Initialize model with correct size
        vocab_size = len(self.processor.card_id_to_index)
        self.model = DynamicDeckClassifier(
            card_vocab_size=vocab_size,
            num_classes=len(self.processor.archetypes)
        )

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        print(f"Model loaded with {vocab_size} unique cards and {len(self.processor.archetypes)} archetypes")


class QuickClashPredictor:
    """Simple interface for making predictions with a trained model"""

    def __init__(self, model_path="clash_royale_classifier.pth"):
        self.trainer = ArchetypeTrainer()
        self.trainer.load_model(model_path)

    def predict_from_url(self, url: str):
        """Predict archetype from Clash Royale deck URL"""
        deck = self.trainer.processor.extract_deck_from_url(url)
        if deck:
            return self.trainer.predict_deck(deck)
        else:
            return {"error": "Could not extract deck from URL"}

    def predict_from_deck_string(self, deck_string: str):
        """Predict archetype from semicolon-separated deck string"""
        deck = self.trainer.processor.extract_from_deck_string(deck_string)
        if deck:
            return self.trainer.predict_deck(deck)
        else:
            return {"error": "Invalid deck string"}

    def predict_from_card_ids(self, card_ids: List[int]):
        """Predict archetype from list of card IDs"""
        if len(card_ids) != 8:
            return {"error": "Deck must contain exactly 8 cards"}
        return self.trainer.predict_deck(card_ids)


def train_new_model():
    """Function to train a new model with your deck data"""
    trainer = ArchetypeTrainer()

    # YOUR TRAINING DATA GOES HERE
    # Replace this with your 100 deck URLs and their archetypes
    training_data = [
        {
            'url': 'https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000010;26000024;26000044;26000050;26000061;26000084;28000000;28000015&l=Royals&tt=159000000',
            'archetype': 'control'
        },
        {
            'url': 'https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000000;26000001;26000010;26000084;27000006;27000008;28000000;28000011&l=Royals&tt=159000000',
            'archetype': 'siege'
        },
        {
            'url': 'https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000010;26000021;26000064;26000065;26000084;27000000;28000014;28000015&l=Royals&tt=159000000',
            'archetype': 'cycle'
        },
        {
            'url': 'https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000010;26000030;26000059;26000072;27000000;28000011;28000014;28000018&l=Royals&tt=159000000',
            'archetype': 'cycle'
        },
        {
            'url': 'https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000010;26000023;26000050;26000059;26000084;27000001;28000007;28000015&l=Royals&tt=159000000',
            'archetype': 'control'
        },
        {
            'url': 'https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000010;26000014;26000021;26000030;26000038;27000000;28000000;28000011&l=Royals&tt=159000000',
            'archetype': 'cycle'
        },
        {
            'url': 'https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000040;26000050;26000053;26000056;26000058;27000004;28000015;28000026&l=Royals&tt=159000000',
            'archetype': 'bait'
        },
        {
            'url': 'https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000006;26000010;26000014;26000032;26000038;27000004;28000015;28000017&l=Royals&tt=159000000',
            'archetype': 'control'
        },
        {
            'url': 'https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000010;26000014;26000021;26000030;26000038;27000000;28000000;28000011&l=Royals&tt=159000000',
            'archetype': 'cycle'
        },
        {
            'url': 'https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000047;26000052;26000057;26000059;27000012;28000000;28000001;28000015&l=Royals&tt=159000000',
            'archetype': 'split_lane'
        },
    ]

    print("Training new model...")
    trainer.load_training_data(training_data)
    trainer.train(epochs=100)
    trainer.save_model("clash_royale_classifier.pth")
    print("Training completed! Model saved as 'clash_royale_classifier.pth'")


def predict_example():
    """Example of how to use the trained model"""
    predictor = QuickClashPredictor("clash_royale_classifier.pth")

    # Example prediction using a URL
    result = predictor.predict_from_url(
        "https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000000;26000001;26000010;26000084;27000006;27000008;28000000;28000011&l=Royals&tt=159000000"
    )

    if 'error' not in result:
        print(f"Deck Archetype: {result['archetype']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Card Types: {result['card_types']}")
        print("\nAll probabilities:")
        for arch, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {arch}: {prob:.2%}")
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    print("Clash Royale Archetype Classifier")
    print("=" * 40)

    # Check if we should train or predict
    response = input("Do you want to (t)rain a new model or (p)redict with existing? [t/p]: ").lower()

    if response == 't':
        print("\nTraining new model...")
        print("NOTE: You need to add your training data to the 'train_new_model()' function first!")
        train_new_model()
    elif response == 'p':
        try:
            predict_example()
        except FileNotFoundError:
            print("Model file not found! You need to train a model first.")
        except Exception as e:
            print(f"Error during prediction: {e}")
    else:
        print("Invalid choice. Please run again and choose 't' or 'p'.")