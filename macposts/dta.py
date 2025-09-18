"""Dynamic Traffic Assignment (DTA).

This module contains two classes for accessing the dynamic traffic assignment
functionalities in macposts. One is for single class DTA and the other is for
biclass DTA.

Note that this module is supposed to serve as an intermediate module between
the core `libmacposts' and the user.

"""

from collections.abc import Iterable

import _macposts_ext as _ext

from .backends import get_backend_state, to_output_array


# XXX: I would like to use a common base class instead.
class _CommonMixin:
    """Mixin class for common methods of both Dta and Mcdta."""

    @classmethod
    def from_files(cls, directory):
        """Create an instance of *cls* with files in *directory*."""
        obj = cls()
        obj.initialize(str(directory))
        return obj

    def register_links(self, links=None):
        """Register *links* for recording cumulative curves.

        If a link is not registered, in order to save memory space, the
        cumulative curves for it will not be available after simulation.

        Note that *links* defaults to None, which means all links will be
        registered.

        """
        if links is None:
            links = self.links
        super().register_links(links)

    def _get_ccs(self, link_func, links):
        """Retrieve cumulative curves using the active array backend."""

        if links is None:
            link_ids = tuple(self.registered_links)
        elif isinstance(links, Iterable) and not isinstance(links, (str, bytes)):
            link_ids = tuple(links)
        else:
            link_ids = (links,)

        state = get_backend_state()
        xp = state.backend.module

        num_rows = int(self.get_cur_loading_interval()) + 1
        num_cols = len(link_ids)
        ccs = xp.full((num_rows, num_cols), xp.nan, dtype=xp.float64)

        for col, link in enumerate(link_ids):
            cc = xp.asarray(link_func(link))
            if cc.size == 0:
                continue
            ticks = cc[:, 0].astype(xp.int64, copy=False)
            values = cc[:, 1].astype(xp.float64, copy=False)
            ccs[ticks, col] = values

        mask = xp.isnan(ccs)
        if state.backend.uses_gpu:
            has_missing = bool(mask.any().item())
        else:
            has_missing = bool(mask.any())
        if has_missing:
            # Forward fill NaNs to keep curves continuous. The implementation is
            # expressed purely in terms of the active array backend so it maps
            # naturally to both NumPy and CuPy.
            row_idx = xp.arange(num_rows, dtype=xp.int64)[:, None]
            idxs = xp.where(~mask, row_idx, 0)
            idxs = xp.maximum.accumulate(idxs, axis=0)
            ccs[mask] = ccs[idxs[mask], xp.nonzero(mask)[1]]

        return to_output_array(ccs, state)


class Dta(_CommonMixin, _ext.Dta):
    """Single class DTA."""

    def get_in_ccs(self, links=None):
        """Get the incoming cumulative curves for registered links.

        Required arguments *links* should be an iterable of link IDs and
        specify the desired links for which the cumulative curves will be
        retrieved. It could also be None, in which case all registered links
        will be used. For backward compatibility, if *links* is not iterable,
        it will be treated as a list of one element. However, that is not
        recommended.

        Return an array of shape (CURRENT-INTERVAL, NUM-LINKS). By default this
        is a NumPy array; when GPU arrays are requested via
        :func:`macposts.backends.configure_array_backend` the result may stay on
        the device.

        """
        return self._get_ccs(self.get_link_in_cc, links)

    def get_out_ccs(self, links=None):
        """Get the outgoing cumulative curves for registered links.

        Required arguments *links* should be an iterable of link IDs and
        specify the desired links for which the cumulative curves will be
        retrieved. It could also be None, in which case all registered links
        will be used. For backward compatibility, if *links* is not iterable,
        it will be treated as a list of one element. However, that is not
        recommended.

        Return an array of shape (CURRENT-INTERVAL, NUM-LINKS). By default this
        is a NumPy array; when GPU arrays are requested via
        :func:`macposts.backends.configure_array_backend` the result may stay on
        the device.

        """
        return self._get_ccs(self.get_link_out_cc, links)


class Mcdta(_CommonMixin, _ext.Mcdta):
    """Biclass DTA."""

    def get_car_in_ccs(self, links=None):
        """Get the incoming car cumulative curves for registered links.

        Required arguments *links* should be an iterable of link IDs and
        specify the desired links for which the cumulative curves will be
        retrieved. It could also be None, in which case all registered links
        will be used.

        Return an array of shape (CURRENT-INTERVAL, NUM-LINKS). By default this
        is a NumPy array; when GPU arrays are requested via
        :func:`macposts.backends.configure_array_backend` the result may stay on
        the device.

        """
        return self._get_ccs(self.get_car_link_in_cc, links)

    def get_car_out_ccs(self, links=None):
        """Get the outgoing car cumulative curves for registered links.

        Required arguments *links* should be an iterable of link IDs and
        specify the desired links for which the cumulative curves will be
        retrieved. It could also be None, in which case all registered links
        will be used.

        Return an array of shape (CURRENT-INTERVAL, NUM-LINKS). By default this
        is a NumPy array; when GPU arrays are requested via
        :func:`macposts.backends.configure_array_backend` the result may stay on
        the device.

        """
        return self._get_ccs(self.get_car_link_out_cc, links)

    def get_truck_in_ccs(self, links=None):
        """Get the incoming truck cumulative curves for registered links.

        Required arguments *links* should be an iterable of link IDs and
        specify the desired links for which the cumulative curves will be
        retrieved. It could also be None, in which case all registered links
        will be used.

        Return an array of shape (CURRENT-INTERVAL, NUM-LINKS). By default this
        is a NumPy array; when GPU arrays are requested via
        :func:`macposts.backends.configure_array_backend` the result may stay on
        the device.

        """
        return self._get_ccs(self.get_truck_link_in_cc, links)

    def get_truck_out_ccs(self, links=None):
        """Get the outgoing truck cumulative curves for registered links.

        Required arguments *links* should be an iterable of link IDs and
        specify the desired links for which the cumulative curves will be
        retrieved. It could also be None, in which case all registered links
        will be used.

        Return an array of shape (CURRENT-INTERVAL, NUM-LINKS). By default this
        is a NumPy array; when GPU arrays are requested via
        :func:`macposts.backends.configure_array_backend` the result may stay on
        the device.

        """
        return self._get_ccs(self.get_truck_link_out_cc, links)


class Mmdta(_CommonMixin, _ext.Mmdta):
    """Multi-modal DTA."""
