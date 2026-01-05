import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Literal, Optional, Union, Tuple, Any, Callable
import functools


class FreezableModule(nn.Module):
    """
    A module that allows selective freezing of parameters using masks.

    This class manages a single mask per parameter that can be updated cumulatively
    through various operations.
    """

    def __init__(self):
        super().__init__()
        # Store a single mask per parameter: {param_name: {"mask": tensor, "hook": hook}}
        self._param_masks = {}

    def _get_parameter(self, param_name: str) -> torch.nn.Parameter:
        """Get a parameter by name."""
        for name, param in self.named_parameters():
            if name == param_name:
                return param
        raise ValueError(f"Parameter '{param_name}' not found")

    def _initialize_mask_if_needed(self, param_name: str) -> None:
        """Initialize a parameter mask if it doesn't exist yet."""
        if param_name not in self._param_masks:
            param = self._get_parameter(param_name)
            # Initialize with all False (nothing frozen)
            self._param_masks[param_name] = {
                "mask": torch.zeros_like(param, dtype=torch.bool),
                "hook": None,
            }

    def _update_hook(self, param_name: str) -> None:
        """Update or create the gradient hook for a parameter based on its current mask."""
        param = self._get_parameter(param_name)
        mask_data = self._param_masks[param_name]

        # Remove existing hook if present
        if mask_data["hook"] is not None:
            mask_data["hook"].remove()
            mask_data["hook"] = None

        # If mask has any True values, create a new hook
        if mask_data["mask"].any():

            def hook_fn(grad, mask=mask_data["mask"]):
                return grad * (~mask.to(grad.device))

            mask_data["hook"] = param.register_hook(functools.partial(hook_fn))

    def update_freeze_mask(
        self,
        param_name: str,
        new_mask: torch.Tensor,
        operation: Literal["set", "add", "remove"] = "set",
    ) -> None:
        """
        Update a parameter's freeze mask.

        Args:
            param_name: Name of the parameter to update
            new_mask: Boolean mask to apply (True = frozen)
            operation: How to combine with existing mask:
                       - "set": Replace the current mask with new_mask
                       - "add": Add to the current mask (freeze more elements)
                       - "remove": Remove from the current mask (unfreeze elements)
        """
        # Get parameter and verify mask shape
        param = self._get_parameter(param_name)
        if new_mask.shape != param.shape:
            raise ValueError(
                f"Mask shape {new_mask.shape} doesn't match parameter shape {param.shape}"
            )

        # Initialize mask if needed
        self._initialize_mask_if_needed(param_name)

        # Update the mask according to the operation
        if operation == "set":
            self._param_masks[param_name]["mask"] = new_mask.bool()
        elif operation == "add":
            self._param_masks[param_name]["mask"] |= new_mask.bool()
        elif operation == "remove":
            self._param_masks[param_name]["mask"] &= ~new_mask.bool()
        else:
            raise ValueError(f"Unknown operation: {operation}")

        # Update the hook
        self._update_hook(param_name)

    def get_freeze_mask(self, param_name: str) -> torch.Tensor:
        """Get the current freeze mask for a parameter."""
        self._initialize_mask_if_needed(param_name)
        return self._param_masks[param_name]["mask"]

    def freeze_parameter(self, param_name: str) -> None:
        """Freeze an entire parameter."""
        param = self._get_parameter(param_name)
        mask = torch.ones_like(param, dtype=torch.bool)
        self.update_freeze_mask(param_name, mask, "set")

    def unfreeze_parameter(self, param_name: str) -> None:
        """Unfreeze an entire parameter."""
        param = self._get_parameter(param_name)
        mask = torch.zeros_like(param, dtype=torch.bool)
        self.update_freeze_mask(param_name, mask, "set")

    def freeze_indices(self, param_name: str, indices: List[int], dim: int = 0) -> None:
        """
        Freeze specific indices of a parameter along a dimension.

        Args:
            param_name: The name of the parameter
            indices: List of indices to freeze
            dim: Dimension along which to index (default: 0)
        """
        param = self._get_parameter(param_name)
        mask = torch.zeros_like(param, dtype=torch.bool)

        # Apply indices to create mask
        if dim == 0:
            mask[indices, ...] = True
        elif dim == 1:
            mask[:, indices, ...] = True
        else:
            slices = [slice(None)] * mask.dim()
            slices[dim] = indices
            mask[tuple(slices)] = True

        self.update_freeze_mask(param_name, mask, "add")

    def unfreeze_indices(
        self, param_name: str, indices: List[int], dim: int = 0
    ) -> None:
        """
        Unfreeze specific indices of a parameter along a dimension.

        Args:
            param_name: The name of the parameter
            indices: List of indices to unfreeze
            dim: Dimension along which to index (default: 0)
        """
        param = self._get_parameter(param_name)
        mask = torch.zeros_like(param, dtype=torch.bool)

        # Apply indices to create mask
        if dim == 0:
            mask[indices, ...] = True
        elif dim == 1:
            mask[:, indices, ...] = True
        else:
            slices = [slice(None)] * mask.dim()
            slices[dim] = indices
            mask[tuple(slices)] = True

        self.update_freeze_mask(param_name, mask, "remove")

    def freeze_all_parameters(self) -> None:
        """Freeze all parameters in the model."""
        for name, _ in self.named_parameters():
            self.freeze_parameter(name)

    def unfreeze_all_parameters(self) -> None:
        """Unfreeze all parameters in the model."""
        for name, _ in self.named_parameters():
            self.unfreeze_parameter(name)

    def freeze_parameters_by_pattern(self, pattern: str) -> None:
        """
        Freeze parameters that match a name pattern.

        Args:
            pattern: String pattern to match parameter names (e.g., 'weight')
                    Uses simple string contains matching
        """
        for name, _ in self.named_parameters():
            if pattern in name:
                self.freeze_parameter(name)

    def unfreeze_parameters_by_pattern(self, pattern: str) -> None:
        """
        Unfreeze parameters that match a name pattern.

        Args:
            pattern: String pattern to match parameter names (e.g., 'weight')
                    Uses simple string contains matching
        """
        for name, _ in self.named_parameters():
            if pattern in name:
                self.unfreeze_parameter(name)

    def get_frozen_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get a dictionary of parameter names and their freeze masks.

        Returns:
            Dictionary mapping parameter names to their freeze masks
        """
        result = {}
        for name in self._param_masks:
            mask = self._param_masks[name]["mask"]
            if mask.any():
                result[name] = mask
        return result

    def get_freeze_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed freeze status information.

        Returns:
            Dictionary with parameter names and freeze statistics
        """
        result = {}
        for name, mask_data in self._param_masks.items():
            mask = mask_data["mask"]
            if mask.any():
                param = self._get_parameter(name)
                total_elements = param.numel()
                frozen_elements = mask.sum().item()
                result[name] = {
                    "total_elements": total_elements,
                    "frozen_elements": frozen_elements,
                    "percent_frozen": (frozen_elements / total_elements) * 100,
                }
        return result

    def is_frozen(self, param_name: str) -> bool:
        """Check if a parameter is completely frozen."""
        if param_name not in self._param_masks:
            return False

        mask = self._param_masks[param_name]["mask"]
        return mask.all()

    def is_partially_frozen(self, param_name: str) -> bool:
        """Check if a parameter is partially frozen."""
        if param_name not in self._param_masks:
            return False

        mask = self._param_masks[param_name]["mask"]
        return mask.any() and not mask.all()
