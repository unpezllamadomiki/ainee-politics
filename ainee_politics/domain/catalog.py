"""Catalog data for the supported politicians."""

from __future__ import annotations

from .models import Politician

DEFAULT_POLITICIANS: tuple[Politician, ...] = (
    Politician("Donald Trump", ("Donald Trump", "Trump"), "president OR election OR campaign"),
    Politician("Joe Biden", ("Joe Biden", "Biden"), "president OR white house OR election"),
    Politician("Kamala Harris", ("Kamala Harris", "Harris"), "vice president OR election OR campaign"),
    Politician("Pedro Sanchez", ("Pedro Sanchez",), "prime minister OR government OR parliament"),
    Politician("Alberto Nunez Feijoo", ("Alberto Nunez Feijoo", "Feijoo"), "opposition OR parliament OR election"),
    Politician("Ursula von der Leyen", ("Ursula von der Leyen",), "european commission OR eu OR parliament"),
    Politician("Emmanuel Macron", ("Emmanuel Macron", "Macron"), "president OR government OR parliament"),
    Politician("Giorgia Meloni", ("Giorgia Meloni", "Meloni"), "prime minister OR government OR parliament"),
    Politician("Olaf Scholz", ("Olaf Scholz", "Scholz"), "chancellor OR government OR parliament"),
    Politician("Keir Starmer", ("Keir Starmer", "Starmer"), "prime minister OR labour OR parliament"),
    Politician("Volodymyr Zelenskyy", ("Volodymyr Zelenskyy", "Zelenskyy", "Zelensky"), "president OR war OR government"),
    Politician("Vladimir Putin", ("Vladimir Putin", "Putin"), "president OR kremlin OR government"),
    Politician("Benjamin Netanyahu", ("Benjamin Netanyahu", "Netanyahu"), "prime minister OR government OR war"),
    Politician("Narendra Modi", ("Narendra Modi", "Modi"), "prime minister OR government OR election"),
    Politician("Xi Jinping", ("Xi Jinping",), "president OR communist party OR government"),
    Politician("Javier Milei", ("Javier Milei", "Milei"), "president OR government OR congress"),
    Politician("Gustavo Petro", ("Gustavo Petro", "Petro"), "president OR government OR congress"),
    Politician("Claudia Sheinbaum", ("Claudia Sheinbaum", "Sheinbaum"), "president OR government OR election"),
    Politician(
        "Luiz Inacio Lula da Silva",
        ("Luiz Inacio Lula da Silva", "Lula da Silva", "Lula"),
        "president OR government OR congress",
    ),
)