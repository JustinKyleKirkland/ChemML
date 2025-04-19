import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PlotSettings:
	"""Stores plot appearance settings."""

	x_title: str = "X Axis"
	y_title: str = "Y Axis"
	title_size: int = 12
	tick_size: int = 10
	zoom_factor: float = 1.2


class PlotController:
	"""
	Controller for plotting functionality that separates business logic from UI.

	This controller handles data operations for plotting, including:
	- Creating various plot types
	- Calculating statistics (R-squared)
	- Managing plot settings and appearance
	"""

	def __init__(self):
		"""Initialize the plotting controller."""
		self.logger = logging.getLogger("PlotController")
		self.df = pd.DataFrame()
		self.figure: Optional[plt.Figure] = None
		self.original_xlim: Optional[Tuple[float, float]] = None
		self.original_ylim: Optional[Tuple[float, float]] = None
		self.settings = PlotSettings()

	def update_data(self, df: pd.DataFrame) -> None:
		"""
		Update the controller with new data.

		Args:
		    df: The DataFrame to use for plotting
		"""
		self.df = df

	def create_scatter_plot(
		self,
		x_column: str,
		y_column: str,
		marker_style: str = "o",
		marker_size: int = 5,
		marker_color: str = "red",
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Create a scatter plot with the given parameters.

		Args:
		    x_column: Name of column for x-axis data
		    y_column: Name of column for y-axis data
		    marker_style: Matplotlib marker style
		    marker_size: Size of markers
		    marker_color: Color of markers

		Returns:
		    Tuple of (x_data, y_data) arrays

		Raises:
		    ValueError: If columns don't exist or data is invalid
		"""
		try:
			if x_column not in self.df.columns:
				raise ValueError(f"Column '{x_column}' not found in data")

			if y_column not in self.df.columns:
				raise ValueError(f"Column '{y_column}' not found in data")

			x_data = self.df[x_column].values
			y_data = self.df[y_column].values

			plt.scatter(
				x_data,
				y_data,
				marker=marker_style,
				s=marker_size,
				color=marker_color,
			)

			return x_data, y_data

		except Exception as e:
			self.logger.error(f"Error creating scatter plot: {e}")
			raise

	def add_trend_line(
		self,
		x_data: np.ndarray,
		y_data: np.ndarray,
		line_color: str = "black",
		line_width: int = 2,
		line_style: str = "-",
	) -> None:
		"""
		Add a trend line (line of best fit) to the current plot.

		Args:
		    x_data: Array of x values
		    y_data: Array of y values
		    line_color: Color of the line
		    line_width: Width of the line
		    line_style: Style of the line (-, --, :, -.)

		Raises:
		    ValueError: If data arrays are invalid
		"""
		try:
			if len(x_data) < 2 or len(y_data) < 2:
				raise ValueError("Need at least two data points for a trend line")

			z = np.polyfit(x_data, y_data, 1)
			p = np.poly1d(z)

			plt.plot(
				x_data,
				p(x_data),
				color=line_color,
				linewidth=line_width,
				linestyle=line_style,
			)

		except Exception as e:
			self.logger.error(f"Error adding trend line: {e}")
			raise

	def calculate_r_squared(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
		"""
		Calculate the R-squared value for a linear fit.

		Args:
		    x_data: Array of x values
		    y_data: Array of y values

		Returns:
		    R-squared value (0 to 1)

		Raises:
		    ValueError: If data arrays are invalid
		"""
		try:
			if len(x_data) <= 1 or len(y_data) <= 1:
				return 0

			mean_y = np.mean(y_data)
			ss_tot = np.sum((y_data - mean_y) ** 2)

			if ss_tot == 0:
				return 0

			z = np.polyfit(x_data, y_data, 1)
			p = np.poly1d(z)
			ss_res = np.sum((y_data - p(x_data)) ** 2)

			return 1 - (ss_res / ss_tot)

		except Exception as e:
			self.logger.error(f"Error calculating R-squared: {e}")
			raise

	def update_plot_appearance(self) -> None:
		"""
		Update plot labels and appearance based on current settings.
		"""
		try:
			plt.xlabel(self.settings.x_title, fontsize=self.settings.title_size)
			plt.ylabel(self.settings.y_title, fontsize=self.settings.title_size)
			plt.tick_params(axis="both", which="major", labelsize=self.settings.tick_size)
			plt.draw()

		except Exception as e:
			self.logger.error(f"Error updating plot appearance: {e}")
			raise

	def zoom_plot(self, zoom_in: bool) -> None:
		"""
		Zoom in or out on the current plot.

		Args:
		    zoom_in: True to zoom in, False to zoom out
		"""
		try:
			ax = plt.gca()
			xlim = ax.get_xlim()
			ylim = ax.get_ylim()

			scale_factor = 1 / self.settings.zoom_factor if zoom_in else self.settings.zoom_factor

			# Calculate new limits maintaining the center point
			new_xlim = [
				xlim[0] + (xlim[1] - xlim[0]) * (1 - scale_factor) / 2,
				xlim[1] - (xlim[1] - xlim[0]) * (1 - scale_factor) / 2,
			]
			new_ylim = [
				ylim[0] + (ylim[1] - ylim[0]) * (1 - scale_factor) / 2,
				ylim[1] - (ylim[1] - ylim[0]) * (1 - scale_factor) / 2,
			]

			ax.set_xlim(new_xlim)
			ax.set_ylim(new_ylim)
			plt.draw()

		except Exception as e:
			self.logger.error(f"Error zooming plot: {e}")
			raise

	def reset_zoom(self) -> None:
		"""
		Reset plot zoom to original limits.
		"""
		try:
			if self.original_xlim and self.original_ylim:
				plt.gca().set_xlim(self.original_xlim)
				plt.gca().set_ylim(self.original_ylim)
				plt.draw()

		except Exception as e:
			self.logger.error(f"Error resetting zoom: {e}")
			raise

	def store_original_limits(self) -> None:
		"""
		Store the current plot limits as the original limits.
		"""
		try:
			self.original_xlim = plt.gca().get_xlim()
			self.original_ylim = plt.gca().get_ylim()

		except Exception as e:
			self.logger.error(f"Error storing original limits: {e}")
			raise
