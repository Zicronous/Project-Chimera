# ai/presets/body_preset.py

class BodyPreset:
    # YOU CHANGE THESE DEFAULTS TO TEST DIFFERENT BODY TYPES
    # Options: muscular, lean, chubby, skinny, bulky
    body_type = "muscular"
    # Options: fair, tan, dark, green, blue
    skin_tone = "fair"
    # Options: normal, long, muscular, thin
    legs = "normal"
    # Options: normal, claws, gloves, mechanical
    hands = "normal"

    @classmethod
    def to_prompt_segment(cls) -> str:
        """Converts the hardcoded preset into a prompt string."""
        parts = []
        
        body_map = {
            "muscular": "extremely muscular, defined abs, broad shoulders",
            "lean": "lean athletic build, toned muscles",
            "chubby": "chubby build, soft round features",
            "skinny": "skinny slender build, thin limbs",
            "bulky": "huge bulky frame, massive muscles"
        }
        parts.append(body_map.get(cls.body_type, cls.body_type))
        
        skin_map = {
            "fair": "fair pale skin",
            "tan": "tanned sun-kissed skin",
            "dark": "dark ebony skin",
            "green": "green skin",
            "blue": "blue skin"
        }
        parts.append(skin_map.get(cls.skin_tone, cls.skin_tone))
        
        leg_map = {
            "normal": "well-proportioned legs",
            "long": "extremely long elegant legs",
            "muscular": "powerful muscular legs",
            "thin": "thin slender legs"
        }
        parts.append(leg_map.get(cls.legs, cls.legs))
        
        hand_map = {
            "normal": "detailed hands",
            "claws": "sharp claws instead of hands",
            "gloves": "wearing combat gloves",
            "mechanical": "cybernetic mechanical hands"
        }
        parts.append(hand_map.get(cls.hands, cls.hands))
        
        return ", ".join(parts)