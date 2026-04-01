def guardrail_filter(user_input: str):
    """
    Blocks illegal, unethical, or deceptive financial requests.
    Returns:
        - None if safe
        - String response if blocked
    """

    text = user_input.lower()

    banned_patterns = [
        # Market manipulation
        "manipulate stock",
        "pump and dump",
        "price rigging",
        "artificially inflate",
        "crash stock price",

        # Insider trading
        "insider trading",
        "inside information",
        "trade without being caught",
        "secret financial info",

        # Fraud / deception
        "fake balance sheet",
        "fake financial report",
        "fabricate earnings",
        "false disclosures",

        # Accounting fraud
        "hide losses",
        "cook the books",
        "accounting manipulation",
        "mislead auditors",

        # Explicit deception
        "deceive investors",
        "mislead investors",
        "financial deception",
    ]

    for phrase in banned_patterns:
        if phrase in text:
            return (
                "⚠️ **Request Blocked — Financial Ethics & Law**\n\n"
                "I can’t help with activities that involve fraud, deception, "
                "market manipulation, insider trading, or misleading investors.\n\n"
                "These actions are illegal and unethical under financial regulations.\n\n"
                "**What I *can* help with instead:**\n"
                "• Understanding how markets work (legally)\n"
                "• Learning financial regulations and compliance\n"
                "• Ethical investing strategies\n"
                "• How analysts detect fraud (for education)\n"
                "• Financial statement analysis (legitimate)\n\n"
                "Try rephrasing your question in a lawful, educational way."
            )

    return None
