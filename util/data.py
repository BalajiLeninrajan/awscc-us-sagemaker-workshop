# Python Built-Ins:
from io import BytesIO
import os
from time import sleep
from typing import Callable, Dict, Iterable, Optional
from urllib.request import urlopen
from zipfile import ZipFile

# Local Dependencies:
import botocore
import numpy as np
import pandas as pd
import sagemaker
from sagemaker.feature_store.feature_definition import FeatureDefinition
from sagemaker.feature_store.feature_group import FeatureGroup, FeatureParameter


def transform_df(df: pd.DataFrame) -> pd.DataFrame:
    # Move 'Survived' to front:
    df = df.loc[:, ["Survived"] + [col for col in df.columns if col != "Survived"]]

    # Add record identifier and event timestamp fields required for SageMaker Feature Store:
    df["PassengerID"] = df.index.to_series().apply(lambda num: f"C-{num:08}")
    df["EventTime"] = (pd.Timestamp.utcnow() - pd.DateOffset(years=1)).timestamp()

    return df


def load_sample_data(
    raw_file_path: str,
    fg_s3_uri: str,
    ignore_cols: Iterable[str] = (
        "PassengerId"
    ),
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame] = transform_df,
    feature_group_name: str = "awscc-sm-titanic",
    feature_group_description: str = (
        "Titanic passenger dataset"
        "https://www.kaggle.com/datasets/shubhamgupta012/titanic-dataset"
    ),
    feature_descriptions: Dict[str, str] = {
        "PassengerID": (
            "Unique passenger identifier (dummy added for purpose of SageMaker Feature Store)"
        ),
        "EventTime": "Event/update timestamp (dummy added for purpose of SageMaker Feature Store)",
        "Survived": "Survival status of the passenger (0 = Not Survived, 1 = Survived)",
        "Pclass": "Passenger class (1 = First class, 2 = Second class, 3 = Third class)",
        "Sex": "Gender of the passenger",
        "Age": "Age of the passenger",
        "SibSp": "Number of siblings/spouses aboard the Titanic",
        "Parch": "Number of parents/children aboard the Titanic",
        "Fare": "Fare paid by the passenger",
        "Embarked": " Port of embarkation (0 = Cherbourg, 1 = Queenstown, 2 = Southampton)"
    },
    feature_parameters: Dict[str, Dict[str, str]] = {
        "Source": {
            "passenger": ["Sex", "Age"],
            "relations": ["SibSp", "Parch"],
            "p_trip_details": ["Fare", "Pclass", "Embarked"],
            "survived": ["Survived"],
        },
    },
    fg_record_identifier_field: str = "PassengerID",
    fg_event_timestamp_field: str = "EventTime",
    sagemaker_session: Optional[sagemaker.Session] = None,
) -> None:
    print(f"Loading {raw_file_path}...")
    df = pd.read_csv(raw_file_path)
    print("Transforming dataframe...")
    df.drop(columns=[col for col in ignore_cols], inplace=True)
    df = transform_fn(df)

    print(f"Setting up SageMaker Feature Store feature group: {feature_group_name}")
    if not sagemaker_session:
        sagemaker_session = sagemaker.Session()
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)

    # Pandas defaults string fields to 'object' dtype, which FS type inference doesn't like:
    for col in df:
        if pd.api.types.is_object_dtype(df[col].dtype):
            df[col] = df[col].astype(pd.StringDtype())

    #print(df.info())
    feature_group.load_feature_definitions(data_frame=df)

    feature_group.create(
        s3_uri=fg_s3_uri,
        record_identifier_name=fg_record_identifier_field,
        event_time_feature_name=fg_event_timestamp_field,
        role_arn=sagemaker.get_execution_role(sagemaker_session),
        enable_online_store=True,
        description=feature_group_description,
    )
    wait_for_fg_creation(feature_group)

    ingestion_manager = feature_group.ingest(data_frame=df, max_processes=16, wait=False)

    print("Configuring feature metadata...")
    update_meta_calls = {}
    for feature_name, desc in feature_descriptions.items():
        update_meta_calls[feature_name] = {"description": desc}
    for param_name, spec in feature_parameters.items():
        for param_value, features in spec.items():
            for feature_name in features:
                if feature_name not in update_meta_calls:
                    update_meta_calls[feature_name] = {}
                feature_spec = update_meta_calls[feature_name]
                if param_value is None:
                    if "parameter_removals" not in feature_spec:
                        feature_spec["parameter_removals"] = [param_name]
                    else:
                        feature_spec["parameter_removals"].append(param_name)
                else:
                    if "parameter_additions" not in feature_spec:
                        feature_spec["parameter_additions"] = [
                            FeatureParameter(key=param_name, value=param_value),
                        ]
                    else:
                        feature_spec["parameter_additions"].append(
                            FeatureParameter(key=param_name, value=param_value),
                        )
    for feature_name, feature_spec in update_meta_calls.items():
        feature_group.update_feature_metadata(feature_name, **feature_spec)
        sleep(2)

    print("Ingesting data to SageMaker Feature Store...")
    ingestion_manager.wait()
    ingest_timestamp = pd.Timestamp.now()


    print("Waiting for propagation to offline Feature Store...")
    ingest_wait_period = pd.DateOffset(
        minutes=5,  # Technically can take 15mins, but who has time for that
    )
    sleep(((ingest_timestamp + ingest_wait_period) - pd.Timestamp.now()).seconds)

    print("Done!")
    return feature_group_name


def describe_fg_if_exists(feature_group: FeatureGroup) -> Optional[dict]:
    try:
        return feature_group.describe()
    except botocore.exceptions.ClientError as e:
        if "Not Found" in e.response["Error"]["Message"]:
            return None
        else:
            raise e


def wait_for_fg_creation(feature_group):
    status = feature_group.describe().get("FeatureGroupStatus")
    print(
        f"Waiting for creation of Feature Group {feature_group.name} (Initial status {status})",
        end="",
    )
    while status == "Creating":
        print(".", end="")
        sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    print()
    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group.name}: {status}")
    print(f"Feature Group {feature_group.name} successfully created.")