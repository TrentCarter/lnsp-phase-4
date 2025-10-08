#!/usr/bin/env python3
"""
QUICK CHUNKER - Just paste your text and run!

1. Replace the text between the triple quotes below
2. Run: python tools/quick_chunk.py
3. See your chunks!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.chunker_v2 import SemanticChunker


# ============================================================================
# 👇👇👇 PASTE YOUR TEXT HERE 👇👇👇
# ============================================================================

MY_TEXT = """
Photosynthesis is the process by which plants convert light energy into chemical energy.
This occurs in the chloroplasts of plant cells. The process requires sunlight, water, and
carbon dioxide as inputs. During photosynthesis, plants absorb light energy through
chlorophyll molecules. This energy is used to convert carbon dioxide and water into
glucose and oxygen. The glucose serves as food for the plant, while oxygen is released
as a byproduct.

Cellular respiration is the metabolic process that converts glucose into ATP energy.
This process occurs in the mitochondria of cells. During cellular respiration, glucose
is broken down through glycolysis, the Krebs cycle, and the electron transport chain.
Oxygen is consumed and carbon dioxide is produced as a waste product. This process
provides energy for all cellular activities in living organisms.

The water cycle describes how water moves between Earth's surface and atmosphere.
Water evaporates from oceans, lakes, and rivers due to solar energy. This water vapor
rises into the atmosphere where it cools and condenses to form clouds. Eventually,
the water returns to Earth's surface as precipitation in the form of rain, snow, or hail.
This continuous cycle is essential for distributing water across the planet and
regulating global climate patterns.
"""

# ============================================================================
# 👆👆👆 PASTE YOUR TEXT ABOVE 👆👆👆
# ============================================================================


# Run the chunker
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SEMANTIC CHUNKER - Quick Results")
    print("=" * 80)

    # Initialize chunker
    chunker = SemanticChunker(min_chunk_size=200)

    # Chunk the text
    chunks = chunker.chunk(MY_TEXT)

    # Show results
    print(f"\n📊 Input: {len(MY_TEXT)} chars, {len(MY_TEXT.split())} words")
    print(f"✓ Created {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"{'─' * 80}")
        print(f"CHUNK {i}")
        print(f"{'─' * 80}")
        print(f"{chunk.word_count} words | {chunk.char_count} chars")
        print()
        print(chunk.text.strip())
        print()

    print("=" * 80)
    print(f"✓ Done! Created {len(chunks)} semantic chunks")
    print("=" * 80)
