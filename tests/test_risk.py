from risk import calculate_position


def test_calculate_position_allocation_cap():
    size, deployed = calculate_position(
        symbol="BTC/USDT",
        price=100,
        total_cap=100000,
        stop_loss_pct=0.005,
        confidence=1.0,
    )
    assert round(size, 6) == 400
    assert round(deployed, 6) == 40000


def test_calculate_position_risk_cap():
    size, deployed = calculate_position(
        symbol="BTC/USDT",
        price=100,
        total_cap=100000,
        stop_loss_pct=0.05,
        confidence=0.5,
    )
    assert size < 400
