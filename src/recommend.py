def retention_strategy(prob):
    if prob > 0.75:
        return "Offer discount + personal call"
    elif prob > 0.40:
        return "Send loyalty rewards email"
    else:
        return "No action required"
