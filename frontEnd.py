import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageDraw, ImageFont, ImageTk
import json
import random

# List of cards (feel free to customize)
ALL_CARDS = [
    "Knight", "Archers", "Goblins", "Giant", "P.E.K.K.A", "Mini P.E.K.K.A",
    "Balloon", "Witch", "Wizard", "Hog Rider", "Valkyrie", "Musketeer",
    "Skeleton Army", "Baby Dragon", "Inferno Tower", "Cannon", "Tesla",
    "Fireball", "Arrows", "Zap", "Rage", "Freeze", "Rocket", "Lightning",
    "Barbarians", "Golem", "Prince", "Dark Prince", "Ice Wizard", "Bandit",
    "Mega Knight", "Electro Wizard", "Lumberjack", "Miner"
]

class DeckBuilderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deck Builder (Custom Images)")

        self.deck = []
        self.card_images = {}

        self.generate_images()
        self.create_ui()

    # --------------------------------------------------
    # Generate fake card images using Pillow
    # --------------------------------------------------
    def generate_images(self):
        for card in ALL_CARDS:
            img = Image.new("RGBA", (120, 120), self.random_color())
            draw = ImageDraw.Draw(img)

            # Try to load a font, fallback to default if needed
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            # Center text
            w, h = draw.textsize(card, font=font)
            draw.text(
                ((120 - w) / 2, (120 - h) / 2),
                card,
                fill="white",
                font=font
            )

            # Resize for GUI thumbnails
            img = img.resize((80, 80))
            self.card_images[card] = ImageTk.PhotoImage(img)

    def random_color(self):
        return (
            random.randint(60, 200),
            random.randint(60, 200),
            random.randint(60, 200)
        )

    # --------------------------------------------------
    # Build GUI
    # --------------------------------------------------
    def create_ui(self):
        left = tk.Frame(self.root)
        left.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(left, text="Available Cards", font=("Arial", 14)).pack()

        canvas = tk.Canvas(left, width=200, height=500)
        scrollbar = tk.Scrollbar(left, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Display each card as an image button
        for card in ALL_CARDS:
            btn = tk.Button(scroll_frame,
                            image=self.card_images[card],
                            text=card,
                            compound="top",
                            width=100,
                            command=lambda c=card: self.add_card(c))
            btn.pack(pady=4)

        # -------- Right Panel: Deck --------
        right = tk.Frame(self.root)
        right.pack(side=tk.RIGHT, padx=20, pady=10)

        tk.Label(right, text="Your Deck (8 cards)", font=("Arial", 14)).pack()

        self.deck_frame = tk.Frame(right)
        self.deck_frame.pack()

        tk.Button(right, text="Clear Deck", command=self.clear_deck).pack(pady=5)
        tk.Button(right, text="Export Deck", command=self.export_deck).pack(pady=5)

    # --------------------------------------------------
    # Deck editing
    # --------------------------------------------------
    def add_card(self, card):
        if len(self.deck) >= 8:
            messagebox.showwarning("Deck Full", "You can only have 8 cards.")
            return
        if card in self.deck:
            messagebox.showwarning("Duplicate", "Card already in deck.")
            return

        self.deck.append(card)
        self.update_deck_display()

    def remove_card(self, card):
        if card in self.deck:
            self.deck.remove(card)
            self.update_deck_display()

    def update_deck_display(self):
        for widget in self.deck_frame.winfo_children():
            widget.destroy()

        for card in self.deck:
            btn = tk.Button(self.deck_frame,
                            image=self.card_images[card],
                            text=card,
                            compound="top",
                            width=100,
                            command=lambda c=card: self.remove_card(c))
            btn.pack(side=tk.LEFT, padx=4)

    # --------------------------------------------------
    # Deck save
    # --------------------------------------------------
    def clear_deck(self):
        self.deck = []
        self.update_deck_display()

    def export_deck(self):
        if len(self.deck) != 8:
            messagebox.showwarning("Incomplete Deck", "Deck must have 8 cards.")
            return

        filename = simpledialog.askstring("Save Dec
