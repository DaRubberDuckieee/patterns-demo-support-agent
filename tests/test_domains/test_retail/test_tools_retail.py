import json

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall
from tau2.data_model.tasks import EnvFunctionCall, Task
from tau2.domains.retail.data_model import (
    CreditCard,
    GiftCard,
    Order,
    OrderItem,
    OrderPayment,
    Product,
    RetailDB,
    User,
    UserAddress,
    UserName,
    Variant,
)
from tau2.domains.retail.environment import get_environment
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_env import (
    EnvironmentEvaluator,
    FullDuplexEnvironmentEvaluator,
)


@pytest.fixture
def retail_db() -> RetailDB:
    return RetailDB(
        users={
            "sara_doe_496": User(
                user_id="sara_doe_496",
                name=UserName(first_name="Sara", last_name="Doe"),
                address=UserAddress(
                    address1="123 Main St",
                    address2="Apt 1",
                    city="San Francisco",
                    state="CA",
                    country="USA",
                    zip="94105",
                ),
                email="sara.doe@example.com",
                payment_methods={
                    "credit_card_0000000": CreditCard(
                        source="credit_card",
                        brand="visa",
                        last_four="1234",
                        id="credit_card_0000000",
                    ),
                    "gift_card_0000000": GiftCard(
                        source="gift_card",
                        balance=100.0,
                        id="gift_card_0000000",
                    ),
                },
                orders=["#W0000000"],
            )
        },
        orders={
            "#W0000000": Order(
                order_id="#W0000000",
                user_id="sara_doe_496",
                status="pending",
                items=[
                    OrderItem(
                        name="Classic T-Shirt",
                        product_id="6086499569",
                        item_id="1008292230",
                        price=29.99,
                        options={"size": "M", "color": "blue"},
                    )
                ],
                address=UserAddress(
                    address1="123 Main St",
                    address2="Apt 1",
                    city="San Francisco",
                    state="CA",
                    country="USA",
                    zip="94105",
                ),
                fulfillments=[],
                payment_history=[
                    OrderPayment(
                        transaction_type="payment",
                        amount=29.99,
                        payment_method_id="credit_card_0000000",
                    )
                ],
            )
        },
        products={
            "6086499569": Product(
                product_id="6086499569",
                name="Classic T-Shirt",
                variants={
                    "1008292230": Variant(
                        item_id="1008292230",
                        price=29.99,
                        available=True,
                        options={"size": "M", "color": "blue"},
                    ),
                    "1008292231": Variant(
                        item_id="1008292231",
                        price=29.99,
                        available=True,
                        options={"size": "L", "color": "blue"},
                    ),
                },
            )
        },
    )


@pytest.fixture
def environment(retail_db: RetailDB) -> Environment:
    environment = get_environment(retail_db)
    environment.tools.find_user_id_by_email("sara.doe@example.com")
    return environment


@pytest.fixture
def calculate_call() -> ToolCall:
    return ToolCall(
        id="0",
        name="calculate",
        arguments={"expression": "2 + 2"},
    )


def test_calculate(environment: Environment, calculate_call: ToolCall):
    response = environment.get_response(calculate_call)
    assert response.content == "4.0"


@pytest.fixture
def cancel_pending_order_call() -> ToolCall:
    return ToolCall(
        id="1",
        name="cancel_pending_order",
        arguments={
            "order_id": "#W0000000",
            "reason": "no longer needed",
        },
    )


def test_cancel_pending_order(
    environment: Environment, cancel_pending_order_call: ToolCall
):
    _ = environment.get_response(cancel_pending_order_call)
    # Check order is cancelled successfully
    order = environment.tools.get_order_details(
        cancel_pending_order_call.arguments["order_id"]
    )
    assert order.status == "cancelled"
    assert order.cancel_reason == "no longer needed"
    # Check refund is added to payment history
    assert len(order.payment_history) == 2
    assert order.payment_history[1].transaction_type == "refund"
    assert order.payment_history[1].amount == order.payment_history[0].amount


@pytest.fixture
def exchange_delivered_order_items_call() -> ToolCall:
    return ToolCall(
        id="2",
        name="exchange_delivered_order_items",
        arguments={
            "order_id": "#W0000000",
            "item_ids": ["1008292230"],
            "new_item_ids": ["1008292231"],
            "payment_method_id": "credit_card_0000000",
        },
    )


def test_exchange_delivered_order_items(
    environment: Environment, exchange_delivered_order_items_call: ToolCall
):
    # First set order status to delivered
    order = environment.tools.get_order_details(
        exchange_delivered_order_items_call.arguments["order_id"]
    )
    order.status = "delivered"

    _ = environment.get_response(exchange_delivered_order_items_call)
    # Check order status is updated
    order = environment.tools.get_order_details(
        exchange_delivered_order_items_call.arguments["order_id"]
    )
    assert order.status == "exchange requested"
    assert order.exchange_items == ["1008292230"]
    assert order.exchange_new_items == ["1008292231"]
    assert order.exchange_payment_method_id == "credit_card_0000000"


@pytest.fixture
def find_user_id_by_name_zip_call() -> ToolCall:
    return ToolCall(
        id="3",
        name="find_user_id_by_name_zip",
        arguments={
            "first_name": "Sara",
            "last_name": "Doe",
            "zip": "94105",
        },
    )


def test_find_user_id_by_name_zip(
    environment: Environment, find_user_id_by_name_zip_call: ToolCall
):
    response = environment.get_response(find_user_id_by_name_zip_call)
    assert response.content == "sara_doe_496"
    # Test with non-existent user
    find_user_id_by_name_zip_call.arguments["first_name"] = "John"
    response = environment.get_response(find_user_id_by_name_zip_call)
    assert response.content == "Error: User not found"


@pytest.fixture
def find_user_id_by_email_call() -> ToolCall:
    return ToolCall(
        id="4",
        name="find_user_id_by_email",
        arguments={"email": "sara.doe@example.com"},
    )


def test_find_user_id_by_email(
    environment: Environment, find_user_id_by_email_call: ToolCall
):
    response = environment.get_response(find_user_id_by_email_call)
    assert response.content == "sara_doe_496"
    # Test with non-existent email
    find_user_id_by_email_call.arguments["email"] = "nonexistent@example.com"
    response = environment.get_response(find_user_id_by_email_call)
    assert response.content == "Error: User not found"


@pytest.fixture
def get_order_details_call() -> ToolCall:
    return ToolCall(
        id="5",
        name="get_order_details",
        arguments={"order_id": "#W0000000"},
    )


def test_get_order_details(environment: Environment, get_order_details_call: ToolCall):
    response = environment.get_response(get_order_details_call)
    # Test with non-existent order
    get_order_details_call.arguments["order_id"] = "#NONEXISTENT"
    response = environment.get_response(get_order_details_call)
    assert response.content == "Error: Order not found"


@pytest.fixture
def get_product_details_call() -> ToolCall:
    return ToolCall(
        id="6",
        name="get_product_details",
        arguments={"product_id": "6086499569"},
    )


def test_get_product_details(
    environment: Environment, get_product_details_call: ToolCall
):
    response = environment.get_response(get_product_details_call)
    # Test with non-existent product
    get_product_details_call.arguments["product_id"] = "NONEXISTENT"
    response = environment.get_response(get_product_details_call)
    assert response.content == "Error: Product not found"


@pytest.fixture
def get_user_details_call() -> ToolCall:
    return ToolCall(
        id="7",
        name="get_user_details",
        arguments={"user_id": "sara_doe_496"},
    )


def test_get_user_details(environment: Environment, get_user_details_call: ToolCall):
    response = environment.get_response(get_user_details_call)
    # Test with non-existent user
    get_user_details_call.arguments["user_id"] = "NONEXISTENT"
    response = environment.get_response(get_user_details_call)
    assert response.content == "Error: Access denied: customer identity does not match"


@pytest.fixture
def list_all_product_types_call() -> ToolCall:
    return ToolCall(
        id="8",
        name="list_all_product_types",
        arguments={},
    )


def test_list_all_product_types(
    environment: Environment, list_all_product_types_call: ToolCall
):
    response = environment.get_response(list_all_product_types_call)
    product_dict = json.loads(response.content)
    assert "Classic T-Shirt" in product_dict


@pytest.fixture
def modify_pending_order_address_call() -> ToolCall:
    return ToolCall(
        id="9",
        name="modify_pending_order_address",
        arguments={
            "order_id": "#W0000000",
            "address1": "456 New St",
            "address2": "Apt 2",
            "city": "San Francisco",
            "state": "CA",
            "country": "USA",
            "zip": "94106",
        },
    )


def test_modify_pending_order_address(
    environment: Environment, modify_pending_order_address_call: ToolCall
):
    response = environment.get_response(modify_pending_order_address_call)
    # Check address is updated
    order = environment.tools.get_order_details(
        modify_pending_order_address_call.arguments["order_id"]
    )
    assert order.address.address1 == "456 New St"
    assert order.address.address2 == "Apt 2"
    assert order.address.zip == "94106"
    # Test with non-pending order
    order.status = "delivered"
    response = environment.get_response(modify_pending_order_address_call)
    assert response.content == "Error: Non-pending order cannot be modified"


@pytest.fixture
def modify_pending_order_items_call() -> ToolCall:
    return ToolCall(
        id="10",
        name="modify_pending_order_items",
        arguments={
            "order_id": "#W0000000",
            "item_ids": ["1008292230"],
            "new_item_ids": ["1008292231"],
            "payment_method_id": "credit_card_0000000",
        },
    )


def test_modify_pending_order_items(
    environment: Environment, modify_pending_order_items_call: ToolCall
):
    response = environment.get_response(modify_pending_order_items_call)
    # Check order is updated
    order = environment.tools.get_order_details(
        modify_pending_order_items_call.arguments["order_id"]
    )
    assert order.status == "pending (item modified)"
    assert order.items[0].item_id == "1008292231"
    # Test with non-pending order
    order.status = "delivered"
    response = environment.get_response(modify_pending_order_items_call)
    assert response.content == "Error: Non-pending order cannot be modified"


@pytest.fixture
def modify_pending_order_payment_call() -> ToolCall:
    return ToolCall(
        id="11",
        name="modify_pending_order_payment",
        arguments={
            "order_id": "#W0000000",
            "payment_method_id": "gift_card_0000000",
        },
    )


def test_modify_pending_order_payment(
    environment: Environment, modify_pending_order_payment_call: ToolCall
):
    response = environment.get_response(modify_pending_order_payment_call)
    # Check payment history is updated
    order = environment.tools.get_order_details(
        modify_pending_order_payment_call.arguments["order_id"]
    )
    assert len(order.payment_history) == 3  # Original payment + new payment + refund
    # Test with non-pending order
    order.status = "delivered"
    response = environment.get_response(modify_pending_order_payment_call)
    assert response.content == "Error: Non-pending order cannot be modified"


@pytest.fixture
def modify_user_address_call() -> ToolCall:
    return ToolCall(
        id="12",
        name="modify_user_address",
        arguments={
            "user_id": "sara_doe_496",
            "address1": "789 New St",
            "address2": "Apt 3",
            "city": "San Francisco",
            "state": "CA",
            "country": "USA",
            "zip": "94107",
        },
    )


def test_modify_user_address(
    environment: Environment, modify_user_address_call: ToolCall
):
    response = environment.get_response(modify_user_address_call)
    # Check address is updated
    user = environment.tools.get_user_details(
        modify_user_address_call.arguments["user_id"]
    )
    assert user.address.address1 == "789 New St"
    assert user.address.address2 == "Apt 3"
    assert user.address.zip == "94107"
    # Test with non-existent user
    modify_user_address_call.arguments["user_id"] = "NONEXISTENT"
    response = environment.get_response(modify_user_address_call)
    assert response.content == "Error: User not found"


@pytest.fixture
def return_delivered_order_items_call() -> ToolCall:
    return ToolCall(
        id="13",
        name="return_delivered_order_items",
        arguments={
            "order_id": "#W0000000",
            "item_ids": ["1008292230"],
            "payment_method_id": "credit_card_0000000",
        },
    )


def test_return_delivered_order_items(
    environment: Environment, return_delivered_order_items_call: ToolCall
):
    # First set order status to delivered
    order = environment.tools.get_order_details(
        return_delivered_order_items_call.arguments["order_id"]
    )
    order.status = "delivered"

    response = environment.get_response(return_delivered_order_items_call)
    # Check order status is updated
    order = environment.tools.get_order_details(
        return_delivered_order_items_call.arguments["order_id"]
    )
    assert order.status == "return requested"
    assert order.return_items == ["1008292230"]
    assert order.return_payment_method_id == "credit_card_0000000"
    # Test with non-delivered order
    order.status = "pending"
    response = environment.get_response(return_delivered_order_items_call)
    assert response.content == "Error: Non-delivered order cannot be returned"


@pytest.fixture
def transfer_to_human_agents_call() -> ToolCall:
    return ToolCall(
        id="15",
        name="transfer_to_human_agents",
        arguments={"summary": "The user wants to cancel the order."},
    )


def test_transfer_to_human_agents(
    environment: Environment, transfer_to_human_agents_call: ToolCall
):
    response = environment.get_response(transfer_to_human_agents_call)
    assert response.content == "Transfer successful"


@pytest.mark.parametrize("last_name", ["Sanchez", "Sánchez", " SÁNCHEZ "])
def test_spoken_identity_keeps_exact_zip(retail_db, last_name):
    from tau2.domains.retail.tools import RetailTools

    retail_db.users["sara_doe_496"].name.last_name = "Sánchez"
    tools = RetailTools(retail_db)
    assert tools.find_user_id_by_name_zip("Sara", last_name, "94105") == "sara_doe_496"
    with pytest.raises(ValueError):
        tools.find_user_id_by_name_zip("Sara", last_name, "00000")


def test_folded_identity_collision_is_not_first_match(retail_db):
    from tau2.domains.retail.tools import RetailTools

    user = retail_db.users["sara_doe_496"]
    user.name.last_name = "Sánchez"
    user.email = "sára@example.com"
    other = user.model_copy(deep=True)
    other.user_id = "other"
    other.name.last_name = "Sanchez"
    other.email = "sara@example.com"
    retail_db.users[other.user_id] = other
    tools = RetailTools(retail_db)
    with pytest.raises(ValueError, match="ambiguous"):
        tools.find_user_id_by_name_zip("Sara", "Sanchez", "94105")
    with pytest.raises(ValueError, match="ambiguous"):
        tools.find_user_id_by_email("SARA@example.com")
    del retail_db.users["other"]
    assert tools.find_user_id_by_email(" SARA@example.com ") == user.user_id


# Include write tools: each returns private data and must reject access before
# either returning that data or mutating an unverified customer's account.
PRIVATE_CALLS = [
    ("get_order_details", {"order_id": "#W0000000"}),
    ("get_user_details", {"user_id": "sara_doe_496"}),
    ("cancel_pending_order", {"order_id": "#W0000000", "reason": "no longer needed"}),
    (
        "modify_pending_order_address",
        dict(
            order_id="#W0000000",
            address1="Other",
            address2="",
            city="SF",
            state="CA",
            country="USA",
            zip="94105",
        ),
    ),
    (
        "modify_user_address",
        dict(
            user_id="sara_doe_496",
            address1="Other",
            address2="",
            city="SF",
            state="CA",
            country="USA",
            zip="94105",
        ),
    ),
    (
        "modify_pending_order_payment",
        {"order_id": "#W0000000", "payment_method_id": "gift_card_0000000"},
    ),
    (
        "modify_pending_order_items",
        dict(
            order_id="#W0000000",
            item_ids=["1008292230"],
            new_item_ids=["1008292231"],
            payment_method_id="credit_card_0000000",
        ),
    ),
    (
        "exchange_delivered_order_items",
        dict(
            order_id="#W0000000",
            item_ids=["1008292230"],
            new_item_ids=["1008292231"],
            payment_method_id="credit_card_0000000",
        ),
    ),
    (
        "return_delivered_order_items",
        dict(
            order_id="#W0000000",
            item_ids=["1008292230"],
            payment_method_id="credit_card_0000000",
        ),
    ),
]


@pytest.mark.parametrize("name,arguments", PRIVATE_CALLS)
@pytest.mark.parametrize("identity", [None, "other"])
def test_private_tools_require_verified_owner(retail_db, name, arguments, identity):
    if "delivered" in name:
        retail_db.orders["#W0000000"].status = "delivered"
    other = retail_db.users["sara_doe_496"].model_copy(deep=True)
    other.user_id = "other"
    other.email = "other@example.com"
    other.orders = []
    retail_db.users["other"] = other
    env = get_environment(retail_db)
    if identity:
        env.tools.find_user_id_by_email(other.email)
    before = env.tools.db.model_dump()
    response = env.get_response(ToolCall(id="private", name=name, arguments=arguments))
    assert response.error
    assert response.content == (
        "Error: Access denied: customer identity does not match"
        if identity
        else "Error: Identity verification required: look up the customer by email or name and zip first"
    )
    assert env.tools.db.model_dump() == before


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("find_user_id_by_email", {"email": "sara.doe@example.com"}),
        (
            "find_user_id_by_name_zip",
            dict(first_name="Sara", last_name="Doe", zip="94105"),
        ),
    ],
)
def test_identity_lookup_unlocks_only_this_conversation(retail_db, name, arguments):
    env = get_environment(retail_db)
    response = env.get_response(ToolCall(id="identity", name=name, arguments=arguments))
    assert not response.error
    for tool_name, tool_args in PRIVATE_CALLS[:2]:
        result = env.get_response(
            ToolCall(id="read", name=tool_name, arguments=tool_args)
        )
        assert not result.error
        assert json.loads(result.content)["user_id"] == "sara_doe_496"
    # A fresh conversation must verify again, even with the same database.
    fresh = get_environment(env.tools.db)
    with pytest.raises(ValueError, match="Identity verification required"):
        fresh.tools.get_order_details("#W0000000")
    env.set_state(None, None, [])
    with pytest.raises(ValueError, match="Identity verification required"):
        env.tools.get_order_details("#W0000000")
    env.tools.find_user_id_by_email("sara.doe@example.com")
    env.tools.update_db()
    with pytest.raises(ValueError, match="Identity verification required"):
        env.tools.get_user_details("sara_doe_496")


@pytest.mark.parametrize("lookup", ["email", "name_zip"])
@pytest.mark.parametrize("ambiguous", [False, True])
def test_failed_lookup_clears_previous_identity(retail_db, lookup, ambiguous):
    env = get_environment(retail_db)
    env.tools.find_user_id_by_email("sara.doe@example.com")
    if ambiguous:
        duplicate = env.tools.db.users["sara_doe_496"].model_copy(deep=True)
        duplicate.user_id = "duplicate"
        env.tools.db.users[duplicate.user_id] = duplicate
    with pytest.raises(ValueError, match="ambiguous" if ambiguous else "not found"):
        if lookup == "email":
            env.tools.find_user_id_by_email(
                "sara.doe@example.com" if ambiguous else "missing@example.com"
            )
        else:
            env.tools.find_user_id_by_name_zip(
                "Sara", "Doe", "94105" if ambiguous else "00000"
            )
    with pytest.raises(ValueError, match="Identity verification required"):
        env.tools.get_order_details("#W0000000")


@pytest.mark.parametrize(
    "lookup,arguments",
    [
        ("find_user_id_by_email", {"email": "sara.doe@example.com"}),
        (
            "find_user_id_by_name_zip",
            dict(first_name="Sara", last_name="Doe", zip="94105"),
        ),
    ],
)
def test_replay_restores_identity_before_order_changes(retail_db, lookup, arguments):
    live = get_environment(retail_db)
    history = []
    for index, (name, args) in enumerate(
        [
            (lookup, arguments),
            (
                "cancel_pending_order",
                {"order_id": "#W0000000", "reason": "no longer needed"},
            ),
        ]
    ):
        call = ToolCall(id=str(index), name=name, arguments=args)
        response = live.get_response(call)
        assert not response.error
        history.extend(
            [AssistantMessage(role="assistant", tool_calls=[call]), response]
        )
    restored = get_environment(retail_db)
    restored.set_state(None, None, history)
    assert restored.tools.get_order_details("#W0000000").status == "cancelled"
    assert restored.tools.db.model_dump() == live.tools.db.model_dump()


def test_public_catalog_remains_available_without_identity(retail_db):
    tools = get_environment(retail_db).tools
    assert tools.get_product_details("6086499569").name == "Classic T-Shirt"
    assert tools.get_item_details("1008292230").available
    assert "Classic T-Shirt" in tools.list_all_product_types()


def test_trusted_setup_does_not_authenticate_the_customer(retail_db):
    env = get_environment(retail_db)
    env.set_state(
        None,
        [
            EnvFunctionCall(
                env_type="assistant",
                func_name="cancel_pending_order",
                arguments={"order_id": "#W0000000", "reason": "no longer needed"},
            )
        ],
        [],
    )
    assert env.tools.db.orders["#W0000000"].status == "cancelled"
    assert "trusted_execution" not in env.tools.get_tools()
    with pytest.raises(ValueError, match="Identity verification required"):
        env.tools.get_order_details("#W0000000")
    # Exceptions and nested scopes must restore the original authorization.
    with pytest.raises(RuntimeError):
        with env.trusted_execution():
            with env.trusted_execution():
                env.tools.find_user_id_by_email("sara.doe@example.com")
            raise RuntimeError("fixture failed")
    with pytest.raises(ValueError, match="Identity verification required"):
        env.tools.get_user_details("sara_doe_496")


@pytest.fixture
def cancellation_task():
    return Task.model_validate(
        {
            "id": "retail-identity",
            "user_scenario": {"instructions": "Cancel my order."},
            "evaluation_criteria": {
                "reward_basis": ["DB"],
                "actions": [
                    {
                        "action_id": "cancel",
                        "name": "cancel_pending_order",
                        "arguments": {
                            "order_id": "#W0000000",
                            "reason": "no longer needed",
                        },
                    }
                ],
            },
        }
    )


@pytest.mark.parametrize(
    "evaluator", [EnvironmentEvaluator, FullDuplexEnvironmentEvaluator]
)
def test_reference_actions_still_require_real_db_changes(
    retail_db, cancellation_task, evaluator
):
    reward = evaluator.calculate_reward(
        environment_constructor=get_environment,
        task=cancellation_task,
        full_trajectory=[],
        env_kwargs={"db": retail_db},
    )
    # Gold actions omit identity lookups. They must still establish the target
    # state so an agent that does nothing cannot receive a passing DB reward.
    assert reward.reward == 0.0
    assert not reward.db_check.db_match


def test_verified_order_change_receives_db_reward(retail_db, cancellation_task):
    env = get_environment(retail_db)
    history = []
    calls = [
        ToolCall(
            id="verify",
            name="find_user_id_by_email",
            arguments={"email": "sara.doe@example.com"},
        ),
        ToolCall(
            id="cancel",
            name="cancel_pending_order",
            arguments={"order_id": "#W0000000", "reason": "no longer needed"},
        ),
    ]
    for call in calls:
        response = env.get_response(call)
        assert not response.error
        history.extend(
            [AssistantMessage(role="assistant", tool_calls=[call]), response]
        )
    reward = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=cancellation_task,
        full_trajectory=history,
        env_kwargs={"db": retail_db},
    )
    assert reward.reward == 1.0
    assert reward.db_check.db_match
