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

# Card database
CLASH_ROYALE_CARDS = {
    26000010: {"name": "Skellies", "elixir": 1, "rarity": 1, "type": "troop"},
    26000084: {"name": "ElectroSpirit", "elixir": 1, "rarity": 1, "type": "troop"},
    26000031: {"name": "FireSpirit", "elixir": 1, "rarity": 1, "type": "troop"},
    26000030: {"name": "IceSpirit", "elixir": 1, "rarity": 1, "type": "troop"},
    26000002: {"name": "Gobs", "elixir": 2, "rarity": 1, "type": "troop"},
    26000019: {"name": "SpearGobs", "elixir": 2, "rarity": 1, "type": "troop"},
    26000013: {"name": "Bomber", "elixir": 2, "rarity": 1, "type": "troop"},
    26000049: {"name": "Bats", "elixir": 2, "rarity": 1, "type": "troop"},
    28000008: {"name": "Zap", "elixir": 2, "rarity": 1, "type": "spell"},
    28000017: {"name": "Snowball", "elixir": 2, "rarity": 1, "type": "spell"},
    26000102: {"name": "Berserker", "elixir": 2, "rarity": 1, "type": "troop"},
    26000001: {"name": "Archers", "elixir": 3, "rarity": 1, "type": "troop"},
    28000001: {"name": "Arrows", "elixir": 3, "rarity": 1, "type": "spell"},
    26000000: {"name": "Knight", "elixir": 3, "rarity": 1, "type": "troop"},
    26000005: {"name": "Minions", "elixir": 3, "rarity": 1, "type": "troop"},
    27000000: {"name": "Cannon", "elixir": 3, "rarity": 1, "type": "building"},
    26000041: {"name": "GobGang", "elixir": 3, "rarity": 1, "type": "troop"},
    26000056: {"name": "SkellyBarrel", "elixir": 3, "rarity": 1, "type": "troop"},
    26000064: {"name": "Firecracker", "elixir": 3, "rarity": 1, "type": "troop"},
    28000018: {"name": "RoyalDelivery", "elixir": 3, "rarity": 1, "type": "spell"},
    26000080: {"name": "SkeletonDragons", "elixir": 4, "rarity": 1, "type": "troop"},
    27000002: {"name": "Mortar", "elixir": 4, "rarity": 1, "type": "building"},
    27000006: {"name": "Tesla", "elixir": 4, "rarity": 1, "type": "building"},
    26000008: {"name": "Barbs", "elixir": 5, "rarity": 1, "type": "troop"},
    26000022: {"name": "Horde", "elixir": 5, "rarity": 1, "type": "troop"},
    26000053: {"name": "Rascals", "elixir": 5, "rarity": 1, "type": "troop"},
    26000024: {"name": "RG", "elixir": 6, "rarity": 1, "type": "troop"},
    26000043: {"name": "eBarbs", "elixir": 6, "rarity": 1, "type": "troop"},
    26000047: {"name": "RoyalRecruits", "elixir": 7, "rarity": 1, "type": "troop"},
    28000016: {"name": "HealSpirit", "elixir": 1, "rarity": 2, "type": "spell"},
    26000038: {"name": "IceGolem", "elixir": 2, "rarity": 2, "type": "troop"},
    26000097: {"name": "SuspiciousBush", "elixir": 2, "rarity": 2, "type": "troop"},
    27000009: {"name": "Tombstone", "elixir": 3, "rarity": 2, "type": "building"},
    26000039: {"name": "MM", "elixir": 3, "rarity": 2, "type": "troop"},
    26000040: {"name": "DartGob", "elixir": 3, "rarity": 2, "type": "troop"},
    28000014: {"name": "Earthquake", "elixir": 3, "rarity": 2, "type": "spell"},
    26000067: {"name": "ElixirGolem", "elixir": 3, "rarity": 2, "type": "troop"},
    28000000: {"name": "Fireball", "elixir": 4, "rarity": 2, "type": "spell"},
    26000018: {"name": "MP", "elixir": 4, "rarity": 2, "type": "troop"},
    26000014: {"name": "Musk", "elixir": 4, "rarity": 2, "type": "troop"},
    27000012: {"name": "GoblinCage", "elixir": 4, "rarity": 2, "type": "building"},
    27000001: {"name": "GobHut", "elixir": 4, "rarity": 2, "type": "building"},
    26000011: {"name": "Valk", "elixir": 4, "rarity": 2, "type": "troop"},
    26000036: {"name": "Ram", "elixir": 4, "rarity": 2, "type": "troop"},
    27000004: {"name": "BombTower", "elixir": 4, "rarity": 2, "type": "building"},
    26000057: {"name": "FlyingMachine", "elixir": 4, "rarity": 2, "type": "troop"},
    26000021: {"name": "Hog", "elixir": 4, "rarity": 2, "type": "troop"},
    26000068: {"name": "BattleHealer", "elixir": 4, "rarity": 2, "type": "troop"},
    27000010: {"name": "Furnace", "elixir": 4, "rarity": 2, "type": "building"},
    26000052: {"name": "Zappies", "elixir": 4, "rarity": 2, "type": "troop"},
    26000095: {"name": "GoblinDemolisher", "elixir": 4, "rarity": 2, "type": "troop"},
    26000003: {"name": "Giant", "elixir": 5, "rarity": 2, "type": "troop"},
    27000003: {"name": "Inferno", "elixir": 5, "rarity": 2, "type": "building"},
    26000017: {"name": "Wiz", "elixir": 5, "rarity": 2, "type": "troop"},
    26000059: {"name": "RoyalHogs", "elixir": 5, "rarity": 2, "type": "troop"},
    28000003: {"name": "Rocket", "elixir": 6, "rarity": 2, "type": "spell"},
    27000005: {"name": "BarbHut", "elixir": 6, "rarity": 2, "type": "building"},
    27000007: {"name": "Pump", "elixir": 6, "rarity": 2, "type": "building"},
    26000028: {"name": "3M", "elixir": 9, "rarity": 2, "type": "troop"},
    28000006: {"name": "Mirror", "elixir": 1.99, "rarity": 3, "type": "spell"},
    28000015: {"name": "BarbBarrel", "elixir": 2, "rarity": 3, "type": "spell"},
    26000058: {"name": "WallBreakers", "elixir": 2, "rarity": 3, "type": "troop"},
    28000024: {"name": "GoblinCurse", "elixir": 2, "rarity": 3, "type": "spell"},
    28000002: {"name": "Rage", "elixir": 2, "rarity": 3, "type": "spell"},
    28000004: {"name": "Barrel", "elixir": 3, "rarity": 3, "type": "spell"},
    26000025: {"name": "Guards", "elixir": 3, "rarity": 3, "type": "troop"},
    26000012: {"name": "Skarmy", "elixir": 3, "rarity": 3, "type": "troop"},
    28000026: {"name": "Vines", "elixir": 3, "rarity": 3, "type": "spell"},
    28000013: {"name": "Clone", "elixir": 3, "rarity": 3, "type": "spell"},
    28000012: {"name": "Tornado", "elixir": 3, "rarity": 3, "type": "spell"},
    28000023: {"name": "Void", "elixir": 3, "rarity": 3, "type": "spell"},
    26000015: {"name": "BabyD", "elixir": 4, "rarity": 3, "type": "troop"},
    26000027: {"name": "DarkPrince", "elixir": 4, "rarity": 3, "type": "troop"},
    28000005: {"name": "Freeze", "elixir": 4, "rarity": 3, "type": "spell"},
    28000009: {"name": "Poison", "elixir": 4, "rarity": 3, "type": "spell"},
    26000101: {"name": "RuneGiant", "elixir": 4, "rarity": 3, "type": "troop"},
    26000044: {"name": "Hunter", "elixir": 4, "rarity": 3, "type": "troop"},
    27000013: {"name": "GoblinDrill", "elixir": 4, "rarity": 3, "type": "building"},
    26000007: {"name": "Witch", "elixir": 5, "rarity": 3, "type": "troop"},
    26000006: {"name": "Balloon", "elixir": 5, "rarity": 3, "type": "troop"},
    26000016: {"name": "Prince", "elixir": 5, "rarity": 3, "type": "troop"},
    26000063: {"name": "eDragon", "elixir": 5, "rarity": 3, "type": "troop"},
    26000034: {"name": "Bowler", "elixir": 5, "rarity": 3, "type": "troop"},
    26000045: {"name": "Exe", "elixir": 5, "rarity": 3, "type": "troop"},
    26000054: {"name": "CannonCart", "elixir": 5, "rarity": 3, "type": "troop"},
    26000020: {"name": "GiantSkelly", "elixir": 6, "rarity": 3, "type": "troop"},
    28000007: {"name": "Lightning", "elixir": 6, "rarity": 3, "type": "spell"},
    26000060: {"name": "GobGiant", "elixir": 6, "rarity": 3, "type": "troop"},
    27000008: {"name": "XBow", "elixir": 6, "rarity": 3, "type": "building"},
    26000004: {"name": "PEKKA", "elixir": 7, "rarity": 3, "type": "troop"},
    26000085: {"name": "ElectroGiant", "elixir": 7, "rarity": 3, "type": "troop"},
    26000009: {"name": "Golem", "elixir": 8, "rarity": 3, "type": "troop"},
    28000011: {"name": "Log", "elixir": 2, "rarity": 4, "type": "spell"},
    26000032: {"name": "Miner", "elixir": 3, "rarity": 4, "type": "troop"},
    26000026: {"name": "Princess", "elixir": 3, "rarity": 4, "type": "troop"},
    26000023: {"name": "IceWiz", "elixir": 3, "rarity": 4, "type": "troop"},
    26000050: {"name": "Ghost", "elixir": 3, "rarity": 4, "type": "troop"},
    26000046: {"name": "Bandit", "elixir": 3, "rarity": 4, "type": "troop"},
    26000061: {"name": "Fisherman", "elixir": 3, "rarity": 4, "type": "troop"},
    26000042: {"name": "eWiz", "elixir": 4, "rarity": 4, "type": "troop"},
    26000037: {"name": "InfernoD", "elixir": 4, "rarity": 4, "type": "troop"},
    26000087: {"name": "Phoenix", "elixir": 4, "rarity": 4, "type": "troop"},
    26000062: {"name": "MagicArcher", "elixir": 4, "rarity": 4, "type": "troop"},
    26000035: {"name": "Lumber", "elixir": 4, "rarity": 4, "type": "troop"},
    26000048: {"name": "NightWitch", "elixir": 4, "rarity": 4, "type": "troop"},
    26000083: {"name": "MotherWitch", "elixir": 4, "rarity": 4, "type": "troop"},
    26000051: {"name": "RamRider", "elixir": 5, "rarity": 4, "type": "troop"},
    28000010: {"name": "Graveyard", "elixir": 5, "rarity": 4, "type": "spell"},
    26000096: {"name": "GoblinMachine", "elixir": 5, "rarity": 4, "type": "troop"},
    26000033: {"name": "Sparky", "elixir": 6, "rarity": 4, "type": "troop"},
    28000025: {"name": "SpiritEmpress", "elixir": 6, "rarity": 4, "type": "spell"},
    26000055: {"name": "MegaKnight", "elixir": 7, "rarity": 4, "type": "troop"},
    26000029: {"name": "Lava", "elixir": 7, "rarity": 4, "type": "troop"},
    26000093: {"name": "LittlePrince", "elixir": 3, "rarity": 5, "type": "troop"},
    26000074: {"name": "GoldenKnight", "elixir": 4, "rarity": 5, "type": "troop"},
    26000069: {"name": "SkeletonKing", "elixir": 4, "rarity": 5, "type": "troop"},
    26000065: {"name": "MightyMiner", "elixir": 4, "rarity": 5, "type": "troop"},
    26000072: {"name": "ArcherQueen", "elixir": 5, "rarity": 5, "type": "troop"},
    26000099: {"name": "Goblinstein", "elixir": 5, "rarity": 5, "type": "troop"},
    26000077: {"name": "Monk", "elixir": 5, "rarity": 5, "type": "troop"},
    26000103: {"name": "BossBandit", "elixir": 6, "rarity": 5, "type": "troop"}
}

# Create a reverse mapping for name to ID lookup
CARD_NAME_TO_ID = {info["name"].lower(): card_id for card_id, info in CLASH_ROYALE_CARDS.items()}


# Helper functions
def get_card_info(card_id):
    """Get card information by ID"""
    return CLASH_ROYALE_CARDS.get(card_id, {"name": "Unknown", "elixir": 0, "rarity": 0, "type": "unknown"})


def find_card_id_by_name(card_name):
    """Find card ID by name (case-insensitive, partial match)"""
    card_name_lower = card_name.lower()

    # Exact match first
    if card_name_lower in CARD_NAME_TO_ID:
        return CARD_NAME_TO_ID[card_name_lower]

    # Partial match
    for name, card_id in CARD_NAME_TO_ID.items():
        if card_name_lower in name or name in card_name_lower:
            return card_id

    return None


def calculate_deck_stats(deck):
    """Calculate deck statistics"""
    total_elixir = 0
    card_details = []

    for card_id in deck:
        card_info = get_card_info(card_id)
        total_elixir += card_info["elixir"]
        card_details.append({
            "id": card_id,
            "name": card_info["name"],
            "elixir": card_info["elixir"],
            "type": card_info["type"],
            "rarity": card_info["rarity"]
        })

    avg_elixir = total_elixir / 8

    # Sort cards by elixir cost for cycle calculation
    sorted_cards = sorted(card_details, key=lambda x: x["elixir"])
    four_card_cycle = sum(card["elixir"] for card in sorted_cards[:4])

    return {
        "average_elixir": avg_elixir,
        "four_card_cycle": four_card_cycle,
        "total_elixir": total_elixir,
        "card_details": card_details
    }


def display_deck_analysis(deck, prediction_result):
    """Display comprehensive deck analysis"""
    stats = calculate_deck_stats(deck)

    print("\n" + "=" * 50)
    print("DECK ANALYSIS")
    print("=" * 50)

    print(f"\nArchetype: {prediction_result['archetype']}")
    print(f"Confidence: {prediction_result['confidence']:.2%}")

    print(f"\nAverage Elixir Cost: {stats['average_elixir']:.2f}")
    print(f"4-Card Cycle Cost: {stats['four_card_cycle']}")
    print(f"Total Deck Cost: {stats['total_elixir']}")

    print(f"\nCard Types: {prediction_result['card_types']}")

    print("\nDeck Composition:")
    print("-" * 40)
    for i, card in enumerate(stats['card_details'], 1):
        rarity_names = {1: "Common", 2: "Rare", 3: "Epic", 4: "Legendary", 5: "Champion"}
        print(
            f"{i}. {card['name']} ({card['elixir']} elixir) - {card['type'].title()} - {rarity_names[card['rarity']]}")

    print("\nAll Archetype Probabilities:")
    print("-" * 30)
    for arch, prob in sorted(prediction_result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {arch}: {prob:.2%}")


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
            result = self.trainer.predict_deck(deck)
            display_deck_analysis(deck, result)
            return result
        else:
            return {"error": "Could not extract deck from URL"}

    def predict_from_deck_string(self, deck_string: str):
        """Predict archetype from semicolon-separated deck string"""
        deck = self.trainer.processor.extract_from_deck_string(deck_string)
        if deck:
            result = self.trainer.predict_deck(deck)
            display_deck_analysis(deck, result)
            return result
        else:
            return {"error": "Invalid deck string"}

    def predict_from_card_ids(self, card_ids: List[int]):
        """Predict archetype from list of card IDs"""
        if len(card_ids) != 8:
            return {"error": "Deck must contain exactly 8 cards"}
        result = self.trainer.predict_deck(card_ids)
        display_deck_analysis(card_ids, result)
        return result

    def predict_from_card_names(self, card_names: List[str]):
        """Predict archetype from list of card names"""
        if len(card_names) != 8:
            return {"error": "Deck must contain exactly 8 cards"}

        card_ids = []
        unknown_cards = []

        for name in card_names:
            card_id = find_card_id_by_name(name)
            if card_id:
                card_ids.append(card_id)
            else:
                unknown_cards.append(name)

        if unknown_cards:
            return {"error": f"Unknown cards: {', '.join(unknown_cards)}"}

        result = self.trainer.predict_deck(card_ids)
        display_deck_analysis(card_ids, result)
        return result


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
        # ... (include all your training data here)
        # Add the rest of your 100 training decks
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
        "https://link.clashroyale.com/en/?clashroyale://copyDeck?deck=26000063;26000015;26000009;26000018;26000068;28000012;28000015;27000007&l=Royals&tt=159000000"
    )

    if 'error' in result:
        print(f"Error: {result['error']}")


def interactive_deck_input():
    """Get deck input interactively from user"""
    print("\nChoose input method:")
    print("1. Enter card names (type each card name)")
    print("2. Enter deck URL")
    print("3. Enter card IDs (semicolon-separated)")

    choice = input("\nEnter your choice (1-3): ").strip()

    predictor = QuickClashPredictor("clash_royale_classifier.pth")

    if choice == "1":
        print("\nEnter 8 card names (one per line):")
        card_names = []
        for i in range(8):
            card_name = input(f"Card {i + 1}: ").strip()
            card_names.append(card_name)

        result = predictor.predict_from_card_names(card_names)
        if 'error' in result:
            print(f"Error: {result['error']}")

    elif choice == "2":
        url = input("\nEnter deck URL: ").strip()
        result = predictor.predict_from_url(url)
        if 'error' in result:
            print(f"Error: {result['error']}")

    elif choice == "3":
        deck_string = input("\nEnter card IDs (semicolon-separated): ").strip()
        result = predictor.predict_from_deck_string(deck_string)
        if 'error' in result:
            print(f"Error: {result['error']}")

    else:
        print("Invalid choice!")


if __name__ == "__main__":
    print("Clash Royale Archetype Classifier")
    print("=" * 50)

    # Check if we should train or predict
    response = input(
        "Do you want to (t)rain a new model, (p)redict with existing, or (i)nteractive input? [t/p/i]: ").lower()

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
    elif response == 'i':
        try:
            interactive_deck_input()
        except FileNotFoundError:
            print("Model file not found! You need to train a model first.")
        except Exception as e:
            print(f"Error during prediction: {e}")
    else:
        print("Invalid choice. Please run again and choose 't', 'p', or 'i'.")