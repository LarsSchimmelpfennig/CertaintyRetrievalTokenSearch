from decimal import Decimal, InvalidOperation
import json, re

def canonical_numeric_key(v):
    """
    Return a canonical string for numeric values (e.g., '18' for 18 or 18.0).
    If v isn't numeric, return None so caller can fallback to a plain string key.
    """
    s = str(v).strip()
    try:
        d = Decimal(s)  # handles '18', '18.0', '0018', '1e1', etc.
    except (InvalidOperation, ValueError):
        return None
    if not d.is_finite():
        return None

    # If it's an integer, make it '18' not '18.0' or '1E+1'
    if d == d.to_integral_value():
        return str(int(d))

    # Otherwise, format without scientific notation and strip trailing zeros
    out = format(d.normalize(), 'f')
    if '.' in out:
        out = out.rstrip('0').rstrip('.')
    return out or '0'

def extract_first_json(text):
    """
    Extracts the first valid JSON object found in the given text.
    
    Args:
        text (str): The input text from which to extract the JSON object.
    
    Returns:
        dict or None: The extracted JSON object as a dictionary if valid, otherwise None.
    """
    json_pattern = re.compile(r"\{.*?\}", re.DOTALL)
    match = json_pattern.search(text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def top_2_delta(output_probs):
    """
    Expectes array of sorted output probs.

    Returns confidence of the LLM by the difference in the probabilities of the top-2 output sequences.
    """
    if len(output_probs) > 1:
        return output_probs[0] - output_probs[1]
    return output_probs[0]


def reorder_past(past, beam_idx, model=None):
    """
    Reorder `past_key_values` across batch dimension to match `beam_idx`.
    Supports:
      - New Cache API (e.g., DynamicCache/StaticCache) via `.reorder_cache` or `.index_select`
      - Legacy tuple-of-tuples PKV
      - Optional fallback to model._reorder_cache if it *is* implemented
    """
    if past is None:
        return None

    # New Cache API (preferred)
    if hasattr(past, "reorder_cache"):
        return past.reorder_cache(beam_idx)
    if hasattr(past, "index_select"):
        # Some Cache implementations expose index_select
        return past.index_select(beam_idx)

    # Optional model hook if present (your install throws NotImplementedError—so we guard)
    if model is not None and hasattr(model, "_reorder_cache"):
        try:
            return model._reorder_cache(past, beam_idx)
        except NotImplementedError:
            pass

    # Legacy PKV: tuple of layers, each (k,v) with batch dim 0
    if isinstance(past, (tuple, list)) and len(past) and isinstance(past[0], (tuple, list)):
        reordered = []
        for layer in past:
            k, v = layer
            reordered.append((k.index_select(0, beam_idx), v.index_select(0, beam_idx)))
        return tuple(reordered)

    raise TypeError("Unsupported past_key_values type; cannot reorder.")