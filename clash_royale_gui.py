# clash_royale_gui.py
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from typing import List, Dict
import os
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests
import json
from collections import Counter
import re
from urllib.parse import unquote
import pickle
import training_data
import cards

# Import your existing classes and functions
from clash_royale_archetype_classifier import *

# Material You Color Palette
COLORS = {
    'primary': '#6750A4',
    'on_primary': '#FFFFFF',
    'primary_container': '#EADDFF',
    'on_primary_container': '#21005D',
    'secondary': '#625B71',
    'on_secondary': '#FFFFFF',
    'secondary_container': '#E8DEF8',
    'on_secondary_container': '#1D192B',
    'surface': '#FFFBFE',
    'on_surface': '#1C1B1F',
    'surface_variant': '#E7E0EC',
    'on_surface_variant': '#49454F',
    'background': '#FFFBFE',
    'on_background': '#1C1B1F',
    'error': '#B3261E',
    'on_error': '#FFFFFF',
    'outline': '#79747E',
    'shadow': '#000000',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'info': '#2196F3'
}


# python
class MaterialButton(tk.Frame):
    def __init__(self, parent, text, command=None, style='filled', width=120, height=36):
        super().__init__(parent, bg=COLORS['background'])
        self.command = command
        self.style = style
        self.text = text
        self._state = 'normal'  # 'normal' or 'disabled'

        # Configure styles
        if style == 'filled':
            self.bg_color = COLORS['primary']
            self.fg_color = COLORS['on_primary']
            self.hover_color = '#5A4791'
        elif style == 'tonal':
            self.bg_color = COLORS['secondary_container']
            self.fg_color = COLORS['on_secondary_container']
            self.hover_color = '#D5C6F0'
        else:  # outlined
            # use surface as transparent substitute for Tk
            self.bg_color = COLORS['surface']
            self.fg_color = COLORS['primary']
            self.hover_color = COLORS['primary_container']

        # Keep originals for restore
        self._orig_bg = self.bg_color
        self._orig_fg = self.fg_color
        self._orig_hover = self.hover_color
        self._orig_cursor = 'hand2'

        self.canvas = tk.Canvas(self, width=width, height=height,
                                bg=COLORS['background'], highlightthickness=0)
        self.canvas.pack()

        # Draw button
        self.rect = self.canvas.create_rectangle(2, 2, width - 2, height - 2,
                                                 fill=self.bg_color, outline='', width=0)
        self.text_id = self.canvas.create_text(width // 2, height // 2, text=text,
                                               fill=self.fg_color, font=('Segoe UI', 10, 'bold'))

        # Bind events
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<Enter>', self.on_enter)
        self.canvas.bind('<Leave>', self.on_leave)
        self.bind('<Button-1>', self.on_click)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

        # Set widget cursor
        self.configure(cursor=self._orig_cursor)

    def on_click(self, event):
        if self._state != 'normal':
            return
        if self.command:
            try:
                self.command()
            except Exception:
                pass

    def on_enter(self, event):
        if self._state != 'normal':
            return
        try:
            self.canvas.itemconfig(self.rect, fill=self.hover_color)
        except Exception:
            pass

    def on_leave(self, event):
        if self._state != 'normal':
            return
        try:
            self.canvas.itemconfig(self.rect, fill=self.bg_color)
        except Exception:
            pass

    def config(self, **kwargs):
        """Accept `state` and `text`, forward other config options to the Frame."""
        state = kwargs.pop('state', None)
        if state is not None:
            self.set_state(state)
        if 'text' in kwargs:
            txt = kwargs.pop('text')
            self.text = txt
            try:
                self.canvas.itemconfig(self.text_id, text=txt)
            except Exception:
                pass
        # forward any other known Frame/Widget options
        if kwargs:
            super().config(**kwargs)

    configure = config  # alias

    def set_state(self, state):
        state = (state or '').lower()
        if state in ('disabled', 'disable', 'off'):
            self._state = 'disabled'
            disabled_bg = '#D3D3D3'
            disabled_fg = '#7F7F7F'
            try:
                self.canvas.itemconfig(self.rect, fill=disabled_bg)
                self.canvas.itemconfig(self.text_id, fill=disabled_fg)
            except Exception:
                pass
            # change cursor to default
            try:
                super().config(cursor='arrow')
            except Exception:
                pass
        else:
            self._state = 'normal'
            self.bg_color = self._orig_bg
            self.fg_color = self._orig_fg
            self.hover_color = self._orig_hover
            try:
                self.canvas.itemconfig(self.rect, fill=self.bg_color)
                self.canvas.itemconfig(self.text_id, fill=self.fg_color)
            except Exception:
                pass
            try:
                super().config(cursor=self._orig_cursor)
            except Exception:
                pass

    def disable(self):
        self.set_state('disabled')

    def enable(self):
        self.set_state('normal')



class MaterialCombobox(ttk.Combobox):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(
            font=('Segoe UI', 9),
            state="readonly"
        )


class MaterialScrollbar(ttk.Scrollbar):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style='Material.Vertical.TScrollbar')


class MaterialCard(tk.Frame):
    def __init__(self, parent, title="", padding=16, **kwargs):
        super().__init__(parent, bg=COLORS['surface'], relief='flat', bd=0, **kwargs)
        self.config(highlightbackground=COLORS['outline'], highlightthickness=1)

        # Title label
        if title:
            title_label = tk.Label(self, text=title, bg=COLORS['surface'],
                                   fg=COLORS['on_surface'], font=('Segoe UI', 14, 'bold'),
                                   anchor='w')
            title_label.pack(fill='x', padx=padding, pady=(padding, 8))

        self.content_frame = tk.Frame(self, bg=COLORS['surface'])
        self.content_frame.pack(fill='both', expand=True, padx=padding, pady=(0, padding))


class ModernScrollableFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, bg=COLORS['background'], highlightthickness=0)

        # Create custom styled scrollbar
        style = ttk.Style()
        style.configure('Material.Vertical.TScrollbar',
                        background=COLORS['surface_variant'],
                        troughcolor=COLORS['background'],
                        bordercolor=COLORS['outline'],
                        arrowcolor=COLORS['on_surface_variant'],
                        relief='flat')

        self.scrollbar = MaterialScrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=COLORS['background'])

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Mouse wheel binding
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def scroll_to_top(self):
        """Scroll to the top of the canvas"""
        self.canvas.yview_moveto(0)


class DragDropCardGUI:
    def __init__(self, parent_frame, on_deck_update_callback):
        self.parent = parent_frame
        self.on_deck_update = on_deck_update_callback
        self.card_images = {}
        self.card_buttons = {}
        self.selected_cards = []
        self.deck_slots = []

        self.setup_card_database()
        self.create_drag_drop_interface()
        self.selected_cards = []

    

    def setup_card_database(self):
        """Setup card database and load images"""
        self.all_cards = []
        for card_id, card_info in CLASH_ROYALE_CARDS.items():
            self.all_cards.append({
                'id': card_id,
                'name': card_info['name'],
                'elixir': card_info['elixir'],
                'type': card_info['type'],
                'rarity': card_info['rarity']
            })

        # Sort cards by elixir cost then name
        self.all_cards.sort(key=lambda x: (x['elixir'], x['name']))

    def create_drag_drop_interface(self):
        """Create the drag and drop interface"""
        # Main container for drag-drop
        main_frame = tk.Frame(self.parent, bg=COLORS['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left side - All cards
        left_card = MaterialCard(main_frame, title="Available Cards", padding=12)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # Search and filter
        self.create_search_filter(left_card.content_frame)

        # Cards scrollable area
        self.create_cards_scrollable(left_card.content_frame)

        # Right side - Deck builder
        right_card = MaterialCard(main_frame, title="Your Deck (0/8)", padding=12)
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.deck_card = right_card

        # Deck slots
        self.create_deck_slots(right_card.content_frame)

        # Deck controls
        self.create_deck_controls(right_card.content_frame)

    def create_search_filter(self, parent):
        """Create search and filter controls"""
        search_frame = tk.Frame(parent, bg=COLORS['surface'])
        search_frame.pack(fill=tk.X, pady=(0, 12))

        # Search entry with modern styling
        tk.Label(search_frame, text="🔍", bg=COLORS['surface'],
                 fg=COLORS['on_surface_variant'], font=('Segoe UI', 12)).pack(side=tk.LEFT, padx=(0, 8))

        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                                     bg=COLORS['surface_variant'], fg=COLORS['on_surface'],
                                     insertbackground=COLORS['on_surface'],
                                     relief='flat', font=('Segoe UI', 10),
                                     width=20)
        self.search_entry.pack(side=tk.LEFT, padx=(0, 16))
        self.search_entry.bind('<KeyRelease>', self.on_search)

        # Filter frame
        filter_frame = tk.Frame(search_frame, bg=COLORS['surface'])
        filter_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Elixir filter
        tk.Label(filter_frame, text="Elixir:", bg=COLORS['surface'],
                 fg=COLORS['on_surface_variant'], font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 4))

        self.elixir_var = tk.StringVar(value="All")
        elixir_combo = MaterialCombobox(filter_frame, textvariable=self.elixir_var,
                                        values=["All", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                        width=6)
        elixir_combo.pack(side=tk.LEFT, padx=(0, 12))
        elixir_combo.bind('<<ComboboxSelected>>', self.on_filter)

        # Type filter
        tk.Label(filter_frame, text="Type:", bg=COLORS['surface'],
                 fg=COLORS['on_surface_variant'], font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 4))

        self.type_var = tk.StringVar(value="All")
        type_combo = MaterialCombobox(filter_frame, textvariable=self.type_var,
                                      values=["All", "Troop", "Spell", "Building"],
                                      width=10)
        type_combo.pack(side=tk.LEFT, padx=(0, 12))
        type_combo.bind('<<ComboboxSelected>>', self.on_filter)

        # Rarity filter
        tk.Label(filter_frame, text="Rarity:", bg=COLORS['surface'],
                 fg=COLORS['on_surface_variant'], font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 4))

        self.rarity_var = tk.StringVar(value="All")
        rarity_combo = MaterialCombobox(filter_frame, textvariable=self.rarity_var,
                                        values=["All", "Common", "Rare", "Epic", "Legendary", "Champion"],
                                        width=10)
        rarity_combo.pack(side=tk.LEFT)
        rarity_combo.bind('<<ComboboxSelected>>', self.on_filter)

    def create_cards_scrollable(self, parent):
        """Create scrollable area for cards"""
        self.scrollable_frame = ModernScrollableFrame(parent)
        self.scrollable_frame.pack(fill=tk.BOTH, expand=True)

        # Load and display cards
        self.load_card_images()
        self.display_cards()

    def load_card_images(self):
        """Load card images from images directory"""
        images_dir = "images"

        if not os.path.exists(images_dir):
            print(f"Warning: Images directory '{images_dir}' not found. Using text buttons.")
            return

        for card in self.all_cards:
            card_name = card['name']
            image_path = os.path.join(images_dir, f"{card_name}.png")

            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                    image = image.resize((90, 110), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)
                    self.card_images[card_name] = photo
                except Exception as e:
                    print(f"Error loading image for {card_name}: {e}")
                    self.card_images[card_name] = None
            else:
                self.card_images[card_name] = None

    def display_cards(self, cards_to_show=None):
        """Display cards in the scrollable area"""
        if cards_to_show is None:
            cards_to_show = self.all_cards

        # Clear existing cards
        for widget in self.scrollable_frame.scrollable_frame.winfo_children():
            widget.destroy()

        self.card_buttons = {}

        # Display cards in grid
        row, col = 0, 0
        max_cols = 5

        for card in cards_to_show:
            card_frame = tk.Frame(self.scrollable_frame.scrollable_frame,
                                  bg=COLORS['surface_variant'], relief='raised',
                                  borderwidth=1, cursor='hand2')
            card_frame.grid(row=row, column=col, padx=4, pady=4, sticky='nsew')
            card_frame.card_data = card

            # Make frame responsive to click
            card_frame.bind('<Button-1>', self.on_card_click)

            if card['name'] in self.card_images and self.card_images[card['name']]:
                # Use image button
                btn = tk.Label(card_frame, image=self.card_images[card['name']],
                               cursor='hand2', bg=COLORS['surface_variant'])
                btn.pack(padx=2, pady=2)
                btn.card_data = card
                btn.bind('<Button-1>', self.on_card_click)

                # Elixir cost overlay
                elixir_label = tk.Label(card_frame, text=str(card['elixir']),
                                        bg=COLORS['primary'], fg=COLORS['on_primary'],
                                        font=('Segoe UI', 10, 'bold'), width=2)
                elixir_label.place(x=2, y=2)
            else:
                # Use text button
                btn_text = f"{card['name']}\n({card['elixir']}⏱️)"
                btn = tk.Label(card_frame, text=btn_text, cursor='hand2',
                               bg=COLORS['surface_variant'], fg=COLORS['on_surface_variant'],
                               font=('Segoe UI', 8), wraplength=80, justify='center')
                btn.pack(padx=8, pady=8)
                btn.card_data = card
                btn.bind('<Button-1>', self.on_card_click)

            self.card_buttons[card['name']] = btn

            col += 1
            if col >= max_cols:
                col = 0
                row += 1

        # Configure grid weights
        for i in range(max_cols):
            self.scrollable_frame.scrollable_frame.columnconfigure(i, weight=1)

    def create_deck_slots(self, parent):
        """Create deck slot areas"""
        slots_frame = tk.Frame(parent, bg=COLORS['surface'])
        slots_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        # Create 2 rows of 4 slots
        for row in range(2):
            for col in range(4):
                slot_num = row * 4 + col
                slot_frame = tk.Frame(slots_frame, bg=COLORS['surface_variant'],
                                      relief='sunken', borderwidth=2,
                                      width=110, height=130, cursor='hand2')
                slot_frame.grid(row=row, column=col, padx=6, pady=6, sticky='nsew')
                slot_frame.grid_propagate(False)

                # Slot label
                slot_label = tk.Label(slot_frame, text=f"Slot {slot_num + 1}",
                                      bg=COLORS['surface_variant'],
                                      fg=COLORS['on_surface_variant'],
                                      font=('Segoe UI', 9))
                slot_label.pack(fill=tk.BOTH, expand=True)

                # Store slot info
                slot_info = {
                    'frame': slot_frame,
                    'label': slot_label,
                    'card': None
                }
                self.deck_slots.append(slot_info)

                # Make clickable
                slot_frame.bind('<Button-1>', lambda e, s=slot_num: self.on_slot_click(s))

        # Configure grid weights
        for i in range(4):
            slots_frame.columnconfigure(i, weight=1)
        for i in range(2):
            slots_frame.rowconfigure(i, weight=1)

    def create_deck_controls(self, parent):
        """Create deck control buttons"""
        controls_frame = tk.Frame(parent, bg=COLORS['surface'])
        controls_frame.pack(fill=tk.X)

        # Clear deck button
        clear_btn = MaterialButton(controls_frame, text="Clear Deck",
                                   command=self.clear_deck, style='tonal')
        clear_btn.pack(side=tk.LEFT, padx=(0, 8))

        # Auto-fill button
        fill_btn = MaterialButton(controls_frame, text="Auto-fill Example",
                                  command=self.auto_fill_example, style='tonal')
        fill_btn.pack(side=tk.LEFT)

        # Deck status
        self.deck_status = tk.Label(controls_frame, text="Deck: 0/8 cards",
                                    bg=COLORS['surface'], fg=COLORS['on_surface_variant'],
                                    font=('Segoe UI', 10))
        self.deck_status.pack(side=tk.RIGHT)

    def on_card_click(self, event):
        """Handle card click - add to first empty deck slot"""
        card_data = event.widget.card_data

        # Check if card is already in deck
        if self.is_card_in_deck(card_data):
            messagebox.showwarning("Duplicate Card", f"{card_data['name']} is already in your deck!")
            return

        # Find first empty slot
        empty_slot = None
        for slot in self.deck_slots:
            if slot['card'] is None:
                empty_slot = slot
                break

        if empty_slot:
            self.add_card_to_slot(card_data, empty_slot)
        else:
            messagebox.showinfo("Deck Full", "Your deck is full! Remove a card first.")

    def is_card_in_deck(self, card_data):
        """Check if a card is already in the deck"""
        for slot in self.deck_slots:
            if slot['card'] is not None and slot['card']['id'] == card_data['id']:
                return True
        return False

    def on_slot_click(self, slot_index):
        """Handle slot click - remove card from slot"""
        slot = self.deck_slots[slot_index]
        if slot['card'] is not None:
            self.remove_card_from_slot(slot)

    def add_card_to_slot(self, card_data, slot):
        """Add card to a deck slot"""
        slot['card'] = card_data

        # Clear slot
        for widget in slot['frame'].winfo_children():
            widget.destroy()

        # Add card to slot
        if card_data['name'] in self.card_images and self.card_images[card_data['name']]:
            card_label = tk.Label(slot['frame'], image=self.card_images[card_data['name']],
                                  bg=COLORS['success'], cursor='hand2')
            card_label.pack(fill=tk.BOTH, expand=True)

            # Elixir cost overlay
            elixir_label = tk.Label(slot['frame'], text=str(card_data['elixir']),
                                    bg=COLORS['primary'], fg=COLORS['on_primary'],
                                    font=('Segoe UI', 10, 'bold'), width=2)
            elixir_label.place(x=2, y=2)
        else:
            card_text = f"{card_data['name']}\n({card_data['elixir']}⏱️)"
            card_label = tk.Label(slot['frame'], text=card_text, bg=COLORS['success'],
                                  fg=COLORS['on_primary'], cursor='hand2',
                                  font=('Segoe UI', 8), wraplength=80, justify='center')
            card_label.pack(fill=tk.BOTH, expand=True)

        card_label.card_data = card_data
        card_label.bind('<Button-1>', lambda e: self.remove_card_from_slot(slot))

        self.update_deck_status()

    def remove_card_from_slot(self, slot):
        """Remove card from deck slot"""
        slot['card'] = None

        # Clear and reset slot
        for widget in slot['frame'].winfo_children():
            widget.destroy()

        slot_label = tk.Label(slot['frame'], text=f"Slot {self.deck_slots.index(slot) + 1}",
                              bg=COLORS['surface_variant'], fg=COLORS['on_surface_variant'],
                              font=('Segoe UI', 9))
        slot_label.pack(fill=tk.BOTH, expand=True)

        self.update_deck_status()

    def update_deck_status(self):
        """Update deck status and notify parent"""
        current_cards = [slot['card'] for slot in self.deck_slots if slot['card'] is not None]
        card_count = len(current_cards)

        # Update deck label
        self.deck_card.config(text=f"Your Deck ({card_count}/8)")
        self.deck_status.config(text=f"Deck: {card_count}/8 cards")

        # Notify parent about deck update
        if self.on_deck_update:
            self.on_deck_update(current_cards)

    def clear_deck(self):
        """Clear all cards from deck"""
        for slot in self.deck_slots:
            if slot['card'] is not None:
                self.remove_card_from_slot(slot)

    def auto_fill_example(self):
        """Auto-fill with an example deck"""
        example_cards = [
            "MegaKnight", "PEKKA", "Bandit", "Ghost",
            "EWiz", "Zap", "Poison", "Tornado"
        ]

        self.clear_deck()

        for i, card_name in enumerate(example_cards):
            if i < len(self.deck_slots):
                # Find card data
                card_data = None
                for card in self.all_cards:
                    if card['name'].lower() == card_name.lower():
                        card_data = card
                        break

                if card_data:
                    self.add_card_to_slot(card_data, self.deck_slots[i])

    def on_search(self, event=None):
        """Handle search filter"""
        self.apply_filters()
        # Scroll to top when searching
        self.scrollable_frame.scroll_to_top()

    def on_filter(self, event=None):
        """Handle filter changes"""
        self.apply_filters()
        # Scroll to top when filtering
        self.scrollable_frame.scroll_to_top()

    def apply_filters(self):
        """Apply search and filters to card display"""
        search_text = self.search_var.get().lower()
        elixir_filter = self.elixir_var.get()
        type_filter = self.type_var.get()
        rarity_filter = self.rarity_var.get()

        filtered_cards = []

        for card in self.all_cards:
            # Search filter
            if search_text and search_text not in card['name'].lower():
                continue

            # Elixir filter
            if elixir_filter != "All" and str(card['elixir']) != elixir_filter:
                continue

            # Type filter
            if type_filter != "All" and card['type'].title() != type_filter:
                continue

            # Rarity filter
            if rarity_filter != "All":
                rarity_names = {1: "Common", 2: "Rare", 3: "Epic", 4: "Legendary", 5: "Champion"}
                card_rarity = rarity_names.get(card['rarity'], "Unknown")
                if card_rarity != rarity_filter:
                    continue

            filtered_cards.append(card)

        self.display_cards(filtered_cards)

    def get_deck_card_names(self):
        """Get list of card names in current deck"""
        return [slot['card']['name'] for slot in self.deck_slots if slot['card'] is not None]


class ClashRoyaleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Clash Royale Archetype Classifier")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS['background'])

        # Initialize predictor
        self.predictor = None
        self.load_model_attempted = False
        self.current_deck_card_ids = []

        # Style configuration
        self.setup_styles()

        # Create main interface
        self.create_main_interface()

        # Try to load model in background
        self.load_model_background()

    def setup_styles(self):
        """Configure modern styles"""
        style = ttk.Style()
        style.theme_use('clam')

        # Configure ttk styles to match Material You
        style.configure('TNotebook', background=COLORS['background'], borderwidth=0)
        style.configure('TNotebook.Tab',
                        background=COLORS['surface_variant'],
                        foreground=COLORS['on_surface_variant'],
                        padding=[20, 8],
                        font=('Segoe UI', 10))
        style.map('TNotebook.Tab',
                  background=[('selected', COLORS['primary'])],
                  foreground=[('selected', COLORS['on_primary'])])

        # Configure Combobox style
        style.configure('TCombobox',
                        fieldbackground=COLORS['surface_variant'],
                        background=COLORS['surface_variant'],
                        foreground=COLORS['on_surface'],
                        selectbackground=COLORS['primary'],
                        selectforeground=COLORS['on_primary'],
                        borderwidth=1,
                        relief='flat')

        style.map('TCombobox',
                  fieldbackground=[('readonly', COLORS['surface_variant'])],
                  selectbackground=[('readonly', COLORS['primary'])],
                  selectforeground=[('readonly', COLORS['on_primary'])])

        # Configure Scrollbar style
        style.configure('Material.Vertical.TScrollbar',
                        background=COLORS['surface_variant'],
                        troughcolor=COLORS['background'],
                        bordercolor=COLORS['outline'],
                        arrowcolor=COLORS['on_surface_variant'],
                        relief='flat')

    def create_main_interface(self):
        """Create the main GUI interface"""
        # Header
        header_frame = tk.Frame(self.root, bg=COLORS['primary'], height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)

        title_label = tk.Label(header_frame, text="Clash Royale Archetype Classifier",
                               bg=COLORS['primary'], fg=COLORS['on_primary'],
                               font=('Segoe UI', 20, 'bold'))
        title_label.pack(side=tk.LEFT, padx=24, pady=20)

        # Status in header
        self.status_label = tk.Label(header_frame, text="Loading model...",
                                     bg=COLORS['primary'], fg=COLORS['on_primary'],
                                     font=('Segoe UI', 10))
        self.status_label.pack(side=tk.RIGHT, padx=24, pady=20)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # Create tabs
        self.create_drag_drop_tab()
        self.create_text_input_tab()
        self.create_results_tab()

    def create_drag_drop_tab(self):
        """Create drag and drop tab"""
        drag_drop_frame = tk.Frame(self.notebook, bg=COLORS['background'])
        self.notebook.add(drag_drop_frame, text="🎴 Deck Builder")

        # Create drag-drop interface
        self.drag_drop_gui = DragDropCardGUI(drag_drop_frame, self.on_deck_update)

        # Predict button for drag-drop
        predict_frame = tk.Frame(drag_drop_frame, bg=COLORS['background'])
        predict_frame.pack(fill=tk.X, pady=16)

        self.drag_predict_btn = MaterialButton(predict_frame,
                                               text="Analyze Deck Archetype",
                                               command=self.predict_from_drag_drop,
                                               style='filled', width=200)
        self.drag_predict_btn.pack()
        self.drag_predict_btn.config(state='disabled')

    def create_text_input_tab(self):
        """Create traditional text input tab"""
        text_frame = tk.Frame(self.notebook, bg=COLORS['background'])
        self.notebook.add(text_frame, text="📝 Text Input")

        # Input method selection
        self.create_input_method_section(text_frame)

    def create_results_tab(self):
        """Create results display tab"""
        results_frame = tk.Frame(self.notebook, bg=COLORS['background'])
        self.notebook.add(results_frame, text="📊 Analysis Results")

        # Results section
        self.create_results_section(results_frame)

        # Training section
        self.create_training_section(results_frame)

    def create_input_method_section(self, parent):
        """Create deck input method selection section"""
        input_card = MaterialCard(parent, title="Text Input Methods", padding=20)
        input_card.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Method selection
        method_frame = tk.Frame(input_card.content_frame, bg=COLORS['surface'])
        method_frame.pack(fill=tk.X, pady=(0, 20))

        self.input_method = tk.StringVar(value="names")

        ttk.Radiobutton(method_frame, text="Card Names",
                        variable=self.input_method, value="names",
                        command=self.on_method_change).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(method_frame, text="Deck URL",
                        variable=self.input_method, value="url",
                        command=self.on_method_change).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(method_frame, text="Card IDs",
                        variable=self.input_method, value="ids",
                        command=self.on_method_change).pack(side=tk.LEFT)

        # Input areas
        self.create_names_input(input_card.content_frame)
        self.create_url_input(input_card.content_frame)
        self.create_ids_input(input_card.content_frame)

        # Predict button
        self.predict_btn = MaterialButton(input_card.content_frame,
                                          text="Predict Archetype",
                                          command=self.predict_archetype,
                                          style='filled', width=180)
        self.predict_btn.pack(pady=(20, 0))
        self.predict_btn.config(state='disabled')

    def create_names_input(self, parent):
        """Create card names input section"""
        self.names_frame = tk.Frame(parent, bg=COLORS['surface'])

        instruction_label = tk.Label(self.names_frame,
                                     text="Enter 8 card names (one per line):",
                                     bg=COLORS['surface'], fg=COLORS['on_surface'],
                                     font=('Segoe UI', 11))
        instruction_label.pack(anchor=tk.W, pady=(0, 12))

        # Card entries frame
        entries_frame = tk.Frame(self.names_frame, bg=COLORS['surface'])
        entries_frame.pack(fill=tk.X)

        self.card_entries = []
        for i in range(8):
            row_frame = tk.Frame(entries_frame, bg=COLORS['surface'])
            row_frame.pack(fill=tk.X, pady=4)

            label = tk.Label(row_frame, text=f"Card {i + 1}:", width=8,
                             bg=COLORS['surface'], fg=COLORS['on_surface'],
                             font=('Segoe UI', 10))
            label.pack(side=tk.LEFT)

            entry = tk.Entry(row_frame, width=30, bg=COLORS['surface_variant'],
                             fg=COLORS['on_surface'], insertbackground=COLORS['on_surface'],
                             relief='flat', font=('Segoe UI', 10))
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
            self.card_entries.append(entry)

            # Add autocomplete
            self.setup_autocomplete(entry)

        # Control buttons
        control_frame = tk.Frame(self.names_frame, bg=COLORS['surface'])
        control_frame.pack(fill=tk.X, pady=(16, 0))

        fill_btn = MaterialButton(control_frame, text="Fill Example Deck",
                                  command=self.fill_example_deck, style='tonal')
        fill_btn.pack(side=tk.LEFT, padx=(0, 12))

        clear_btn = MaterialButton(control_frame, text="Clear All Fields",
                                   command=self.clear_text_fields, style='tonal')
        clear_btn.pack(side=tk.LEFT)

    def create_url_input(self, parent):
        """Create deck URL input section"""
        self.url_frame = tk.Frame(parent, bg=COLORS['surface'])

        instruction_label = tk.Label(self.url_frame,
                                     text="Enter Clash Royale deck URL:",
                                     bg=COLORS['surface'], fg=COLORS['on_surface'],
                                     font=('Segoe UI', 11))
        instruction_label.pack(anchor=tk.W, pady=(0, 12))

        self.url_entry = tk.Entry(self.url_frame, width=80, bg=COLORS['surface_variant'],
                                  fg=COLORS['on_surface'], insertbackground=COLORS['on_surface'],
                                  relief='flat', font=('Segoe UI', 10))
        self.url_entry.pack(fill=tk.X)

        # Example URL
        example_label = tk.Label(self.url_frame,
                                 text="Example: clashroyale://copyDeck?deck=26000063;26000015;...",
                                 bg=COLORS['surface'], fg=COLORS['on_surface_variant'],
                                 font=('Segoe UI', 9))
        example_label.pack(anchor=tk.W, pady=(8, 0))

        # Control buttons
        control_frame = tk.Frame(self.url_frame, bg=COLORS['surface'])
        control_frame.pack(fill=tk.X, pady=(12, 0))

        clear_btn = MaterialButton(control_frame, text="Clear URL",
                                   command=lambda: self.url_entry.delete(0, tk.END),
                                   style='tonal')
        clear_btn.pack(side=tk.LEFT)

    def create_ids_input(self, parent):
        """Create card IDs input section"""
        self.ids_frame = tk.Frame(parent, bg=COLORS['surface'])

        instruction_label = tk.Label(self.ids_frame,
                                     text="Enter 8 card IDs (semicolon-separated):",
                                     bg=COLORS['surface'], fg=COLORS['on_surface'],
                                     font=('Segoe UI', 11))
        instruction_label.pack(anchor=tk.W, pady=(0, 12))

        self.ids_entry = tk.Entry(self.ids_frame, width=80, bg=COLORS['surface_variant'],
                                  fg=COLORS['on_surface'], insertbackground=COLORS['on_surface'],
                                  relief='flat', font=('Segoe UI', 10))
        self.ids_entry.pack(fill=tk.X)

        # Example
        example_label = tk.Label(self.ids_frame,
                                 text="Example: 26000063;26000015;26000009;26000018;26000068;28000012;28000015;27000007",
                                 bg=COLORS['surface'], fg=COLORS['on_surface_variant'],
                                 font=('Segoe UI', 9))
        example_label.pack(anchor=tk.W, pady=(8, 0))

        # Control buttons
        control_frame = tk.Frame(self.ids_frame, bg=COLORS['surface'])
        control_frame.pack(fill=tk.X, pady=(12, 0))

        clear_btn = MaterialButton(control_frame, text="Clear IDs",
                                   command=lambda: self.ids_entry.delete(0, tk.END),
                                   style='tonal')
        clear_btn.pack(side=tk.LEFT)

    def create_results_section(self, parent):
        """Create results display section"""
        results_card = MaterialCard(parent, title="Prediction Results", padding=16)
        results_card.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_card.content_frame,
                                                      height=20,
                                                      wrap=tk.WORD,
                                                      font=('Consolas', 10),
                                                      bg=COLORS['surface'],
                                                      fg=COLORS['on_surface'],
                                                      insertbackground=COLORS['on_surface'],
                                                      relief='flat')
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Make text widget read-only
        self.results_text.config(state=tk.DISABLED)

    def create_training_section(self, parent):
        """Create model training section"""
        training_frame = tk.Frame(parent, bg=COLORS['background'])
        training_frame.pack(fill=tk.X, pady=16)

        # Training button
        self.train_btn = MaterialButton(training_frame,
                                        text="Train New Model",
                                        command=self.train_model,
                                        style='filled')
        self.train_btn.pack(side=tk.LEFT, padx=(0, 12))

        # Progress bar for training
        self.progress = ttk.Progressbar(training_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def setup_autocomplete(self, entry):
        """Setup autocomplete for card name entry"""

        def autocomplete(event):
            current_text = entry.get().lower()
            if not current_text:
                return

            matches = [name for name in CARD_NAME_TO_ID.keys()
                       if name.startswith(current_text)]

            if matches:
                entry.delete(0, tk.END)
                entry.insert(0, matches[0].title())
                entry.select_range(len(current_text), tk.END)

        entry.bind('<Tab>', autocomplete)

    def on_deck_update(self, current_cards):
        """Callback when drag-drop deck is updated"""
        # Enable predict button if deck is complete
        if len(current_cards) == 8:
            self.drag_predict_btn.config(state='normal')
        else:
            self.drag_predict_btn.config(state='disabled')

    def predict_from_drag_drop(self):
        """Predict archetype from drag-drop deck"""
        if not self.predictor:
            messagebox.showerror("Error", "Model not loaded. Please train a model first.")
            return

        card_names = self.drag_drop_gui.get_deck_card_names()

        if len(card_names) != 8:
            messagebox.showwarning("Incomplete Deck", "Please add exactly 8 cards to your deck.")
            return

        # Store the current deck for detailed display
        card_ids = []
        for name in card_names:
            card_id = find_card_id_by_name(name)
            if card_id:
                card_ids.append(card_id)
        self.current_deck_card_ids = card_ids

        # Switch to results tab
        self.notebook.select(2)

        # Show loading
        self.show_loading_message()

        # Run prediction in background
        threading.Thread(target=self._predict_from_names_thread,
                         args=(card_names,), daemon=True).start()

    def clear_text_fields(self):
        """Clear all text input fields"""
        for entry in self.card_entries:
            entry.delete(0, tk.END)

        self.url_entry.delete(0, tk.END)
        self.ids_entry.delete(0, tk.END)

    def on_method_change(self):
        """Show/hide input sections based on selected method"""
        method = self.input_method.get()

        # Hide all frames first
        self.names_frame.pack_forget()
        self.url_frame.pack_forget()
        self.ids_frame.pack_forget()

        # Show selected frame
        if method == "names":
            self.names_frame.pack(fill=tk.X, pady=(10, 0))
        elif method == "url":
            self.url_frame.pack(fill=tk.X, pady=(10, 0))
        elif method == "ids":
            self.ids_frame.pack(fill=tk.X, pady=(10, 0))

    def fill_example_deck(self):
        """Fill with an example deck"""
        example_cards = [
            "Mega Knight", "P.E.K.K.A", "Bandit", "Royal Ghost",
            "Electro Wizard", "Zap", "Poison", "Tornado"
        ]

        for i, card_name in enumerate(example_cards):
            if i < len(self.card_entries):
                self.card_entries[i].delete(0, tk.END)
                self.card_entries[i].insert(0, card_name)

    def load_model_background(self):
        """Load model in background thread"""

        def load_model():
            try:
                self.predictor = QuickClashPredictor("clash_royale_classifier.pth")
                self.root.after(0, self.on_model_loaded, True, "Model loaded successfully!")
            except FileNotFoundError:
                self.root.after(0, self.on_model_loaded, False, "Model file not found. Train a model first.")
            except Exception as e:
                self.root.after(0, self.on_model_loaded, False, f"Error loading model: {str(e)}")

        threading.Thread(target=load_model, daemon=True).start()

    def on_model_loaded(self, success, message):
        """Callback when model loading completes"""
        self.load_model_attempted = True

        if success:
            self.status_label.config(text=message)
            self.predict_btn.config(state='normal')
            self.drag_predict_btn.config(state='normal')
        else:
            self.status_label.config(text=message)
            self.predict_btn.config(state='disabled')
            self.drag_predict_btn.config(state='disabled')

    def predict_archetype(self):
        """Predict archetype based on input method"""
        if not self.predictor:
            messagebox.showerror("Error", "Model not loaded. Please train a model first.")
            return

        method = self.input_method.get()

        try:
            if method == "names":
                self.predict_from_names()
            elif method == "url":
                self.predict_from_url()
            elif method == "ids":
                self.predict_from_ids()
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction: {str(e)}")

    def predict_from_names(self):
        """Predict from card names"""
        card_names = []
        for entry in self.card_entries:
            name = entry.get().strip()
            if name:
                card_names.append(name)

        if len(card_names) != 8:
            messagebox.showwarning("Input Error", "Please enter exactly 8 card names.")
            return

        # Switch to results tab
        self.notebook.select(2)

        # Show loading
        self.show_loading_message()

        # Run prediction in background
        threading.Thread(target=self._predict_from_names_thread,
                         args=(card_names,), daemon=True).start()

    def _predict_from_names_thread(self, card_names):
        """Background thread for name-based prediction"""
        try:
            # Store the current deck for detailed display
            card_ids = []
            for name in card_names:
                card_id = find_card_id_by_name(name)
                if card_id:
                    card_ids.append(card_id)
            self.current_deck_card_ids = card_ids

            result = self.predictor.predict_from_card_names(card_names)
            self.root.after(0, self.display_prediction_result, result)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def predict_from_url(self):
        """Predict from deck URL"""
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Input Error", "Please enter a deck URL.")
            return

        # Switch to results tab
        self.notebook.select(2)

        # Show loading
        self.show_loading_message()

        # Run prediction in background
        threading.Thread(target=self._predict_from_url_thread,
                         args=(url,), daemon=True).start()

    def _predict_from_url_thread(self, url):
        """Background thread for URL-based prediction"""
        try:
            # Store the current deck for detailed display
            deck = self.predictor.trainer.processor.extract_deck_from_url(url)
            self.current_deck_card_ids = deck

            result = self.predictor.predict_from_url(url)
            self.root.after(0, self.display_prediction_result, result)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def predict_from_ids(self):
        """Predict from card IDs"""
        ids_text = self.ids_entry.get().strip()
        if not ids_text:
            messagebox.showwarning("Input Error", "Please enter card IDs.")
            return

        # Switch to results tab
        self.notebook.select(2)

        # Show loading
        self.show_loading_message()

        # Run prediction in background
        threading.Thread(target=self._predict_from_ids_thread,
                         args=(ids_text,), daemon=True).start()

    def _predict_from_ids_thread(self, ids_text):
        """Background thread for ID-based prediction"""
        try:
            # Store the current deck for detailed display
            deck = self.predictor.trainer.processor.extract_from_deck_string(ids_text)
            self.current_deck_card_ids = deck

            result = self.predictor.predict_from_deck_string(ids_text)
            self.root.after(0, self.display_prediction_result, result)
        except Exception as e:
            self.root.after(0, self.display_error, str(e))

    def show_loading_message(self):
        """Show loading message in results area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Analyzing deck... Please wait...")
        self.results_text.config(state=tk.DISABLED)

    def display_prediction_result(self, result):
        """Display prediction results in the text area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        if 'error' in result:
            self.results_text.insert(tk.END, f"Error: {result['error']}")
        else:
            # Get deck stats and card details
            deck = []
            if hasattr(self, 'current_deck_card_ids'):
                deck = self.current_deck_card_ids
            elif 'deck_stats' in result:
                # Extract card IDs from deck stats if available
                deck = [card['id'] for card in result['deck_stats']['card_details']]

            # Calculate deck stats if not already in result
            if 'deck_stats' not in result and deck:
                result['deck_stats'] = calculate_deck_stats(deck)

            # Format the results with all the detailed information
            output = f"Model loaded with {len(self.predictor.trainer.processor.card_id_to_index)} unique cards and {len(self.predictor.trainer.processor.archetypes)} archetypes\n\n"
            output += "=" * 50 + "\n"
            output += "DECK ANALYSIS\n"
            output += "=" * 50 + "\n\n"

            output += f"Archetype: {result['archetype']}\n"
            output += f"Confidence: {result['confidence']:.2%}\n\n"

            # Deck statistics
            if 'deck_stats' in result:
                stats = result['deck_stats']
                output += f"Average Elixir Cost: {stats['average_elixir']:.2f}\n"
                output += f"4-Card Cycle Cost: {stats['four_card_cycle']}\n"
                output += f"Total Deck Cost: {stats['total_elixir']}\n\n"

            # Card type distribution
            card_types = result.get('card_types', {})
            output += f"Card Types: {card_types}\n\n"

            # Deck composition
            output += "Deck Composition:\n"
            output += "-" * 40 + "\n"

            if 'deck_stats' in result and 'card_details' in result['deck_stats']:
                rarity_names = {1: "Common", 2: "Rare", 3: "Epic", 4: "Legendary", 5: "Champion"}
                for i, card in enumerate(result['deck_stats']['card_details'], 1):
                    output += f"{i}. {card['name']} ({card['elixir']} elixir) - {card['type'].title()} - {rarity_names[card['rarity']]}\n"
            else:
                # Fallback: try to get card details from the deck
                if deck:
                    rarity_names = {1: "Common", 2: "Rare", 3: "Epic", 4: "Legendary", 5: "Champion"}
                    for i, card_id in enumerate(deck, 1):
                        card_info = get_card_info(card_id)
                        output += f"{i}. {card_info['name']} ({card_info['elixir']} elixir) - {card_info['type'].title()} - {rarity_names[card_info['rarity']]}\n"

            output += "\n"

            # All probabilities
            output += "All Archetype Probabilities:\n"
            output += "-" * 30 + "\n"
            all_probs = result.get('all_probabilities', {})
            for arch, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                output += f"  {arch}: {prob:.2%}\n"

            self.results_text.insert(tk.END, output)

        self.results_text.config(state=tk.DISABLED)

    def display_error(self, error_message):
        """Display error message in results area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"❌ ERROR:\n{error_message}")
        self.results_text.config(state=tk.DISABLED)

    def train_model(self):
        """Train a new model"""
        if not messagebox.askyesno("Confirm Training",
                                   "Training a new model may take several minutes. Continue?"):
            return

        # Disable predict button and show progress
        self.predict_btn.config(state='disabled')
        self.drag_predict_btn.config(state='disabled')
        self.train_btn.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="Training model...")

        # Run training in background
        threading.Thread(target=self._train_model_thread, daemon=True).start()

    def _train_model_thread(self):
        """Background thread for model training"""
        try:
            train_new_model()
            self.root.after(0, self.on_training_completed, True, "Training completed successfully!")
        except Exception as e:
            self.root.after(0, self.on_training_completed, False, f"Training failed: {str(e)}")

    def on_training_completed(self, success, message):
        """Callback when training completes"""
        self.progress.stop()
        self.train_btn.config(state='normal')
        self.status_label.config(text=message)

        if success:
            # Reload the model
            self.load_model_background()
            messagebox.showinfo("Training Complete", message)
        else:
            messagebox.showerror("Training Failed", message)


def main():
    """Main function to run the GUI"""
    try:
        root = tk.Tk()
        app = ClashRoyaleGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application: {e}")


if __name__ == "__main__":
    main()