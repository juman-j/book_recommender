'''
Module settings

This module defines the Settings class for configuring the application.
'''
import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseSettings

load_dotenv()

# pylint: disable=R0903
class Settings(BaseSettings):
    """
    Settings class for configuring the application.
    """
    main_url: Optional[str] = os.getenv('MAIN_URL')

settings = Settings()
