#!/usr/bin/env python

import pytest

from igniter.registry import Registry


@pytest.fixture
def registry():
    return Registry()


def test_registry_registration(registry):
    @registry('TestClass')
    class TestClass:
        pass

    assert 'TestClass' in registry
    assert registry['TestClass'] == TestClass


def test_registry_duplicate_registration(registry):
    @registry('TestClass')
    class TestClass:
        pass

    with pytest.raises(ValueError):

        @registry('TestClass')
        class DuplicateClass:
            pass


def test_registry_get(registry):
    @registry('TestClass')
    class TestClass:
        pass

    assert registry.get('TestClass') == TestClass
