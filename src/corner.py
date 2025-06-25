import numpy as np
from chainconsumer import ChainConsumer

class AxesCornerPlot():
    """
    Allows easy axis limit and label customisation on top of ChainConsumer
    corner plots. Each parameter has both a tag (a simple string 
    representation) and a label (to use when plotting). If no labels are
    passed, will use the tags as the labels.
    """
    def __init__(self, *tagged_chains : dict, labels=None, param_truths=None, plotter_kwargs={}):
        """
        Allows easy axis limit and label customisation on top of ChainConsumer
        corner plots. Each parameter has both a tag (a simple string 
        representation) and a label (to use when plotting). If no labels are
        passed, will use the tags as the labels.

        Parameters
        ----------
        tagged_chains : dict
            Dictionaries containing the flattened chains with parameter tags as
            keys, and chains as values. Optionally also contains a 'config' key,
            whose value is a dictionary of keyword arguments to be passed to
            ChainConsumer.add_chain(). This can be used e.g. to set the chain
            names.
        labels : list of str, optional
            List containing the labels to plot the corner plot with. Distinct
            from tags to allow easy references to otherwise long, LaTeX markup 
            string labels. If not passed, uses the chain's tags as labels.
        param_truths : list of floats, optional
            List containing the fiducial parameter values.
        plotter_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the ChainConsumer.plotter
            method.
        """
        # Check labels and param_truths lengths are the same as the dimension of 
        # all the tagged chains.
        if labels is not None:
            for tagged_chain in tagged_chains:
                chain_dim = len([key for key in tagged_chain.keys() \
                                 if key!='config'])
                if len(labels) != chain_dim:
                    errmsg = "Number of tagged chain parameters for each chain " \
                           + "should equal number of labels, but lengths are " \
                           + f"{chain_dim} and {len(labels)}"
                    raise ValueError(errmsg)
        else:
            try:
                labels = list(tagged_chains[0].keys())
            except IndexError:
                raise TypeError("No chains were passed.")
        
        if param_truths is not None:
            if len(labels) != len(param_truths):
                errmsg = "Number of tagged chain parameters should equal " \
                       + "number of parameter truths, but lengths are " \
                       + f"{len(tagged_chain)} and {len(param_truths)}"
                raise ValueError(errmsg)

        self.dim    = len(labels)
        self.labels = labels
        self.tags   = tagged_chains[0].keys()

        self.consumer = ChainConsumer()
        for tagged_chain in tagged_chains:
            try:
                chain_kwargs = tagged_chain.pop('config')
            except KeyError:
                chain_kwargs = {}
            self.consumer.add_chain(tagged_chain, **chain_kwargs)
        
        self.cornerfig = self.consumer.plotter.plot(truth=param_truths, **plotter_kwargs)
        self.axiscube  = np.reshape(self.cornerfig.axes, (self.dim, self.dim))

        # Set the labels of the corner plot, overriding the default
        # ChainConsumer behavior of setting the labels to the keys of the chain
        # dictionary passed.
        for tag, label in zip(self.tags, self.labels):
            self.set_label(tag, label)
            try:
                self.set_xticks(tag, ticks='plain')
            except ValueError:
                pass
            try:
                self.set_yticks(tag, ticks='plain')
            except ValueError:
                pass
    
    def _get_row_axes(self, tag):
        """
        Return the row of axes of the parameter corresponding to 'tag' in order
        from left to right.
        """
        tag_index = list(self.tags).index(tag)
        if tag_index == 0:
            raise ValueError(f"element '{tag}' has no row axes.")
        
        return self.axiscube[tag_index, :tag_index]
    
    def _get_column_axes(self, tag):
        """
        Return the column of axes of the parameter corresponding to 'tag' in 
        order from top to bottom.
        """
        tag_index = list(self.tags).index(tag)
        if tag_index == (self.dim-1):
            raise ValueError(f"element '{tag}' has no column axes.")
        
        return self.axiscube[tag_index-(self.dim-1):,tag_index]

    def _get_hist_axes(self, tag):
        """
        Return the histogram of axes of the parameter corresponding to 'tag'.
        """
        tag_index = list(self.tags).index(tag)
        return self.axiscube[tag_index,tag_index]

    def set_tick_params(self, **kwargs):
        """
        (Re)set the tick parameters in the corner plot, e.g. labelsize.
        Keyword arguments are passed to matplotlib.axes.Axes.tick_params. 
        See the documentation for more information.
        """
        x_axes = self.axiscube[-1]
        y_axes = self.axiscube[1:,0]
        for axes in x_axes:
            axes.tick_params(axis='x', **kwargs)
        for axes in y_axes:
            axes.tick_params(axis='y', **kwargs)

    def set_labelpad(self, labelpad):
        """
        Set the padding between the axes and the axes labels.
        """
        x_axes = self.axiscube[-1]
        y_axes = self.axiscube[1:,0]
        tick_label_heights = []
        for axes in x_axes:
            axes.xaxis.set_label_coords(0.5, -labelpad)
        for axes in y_axes:
            axes.yaxis.set_label_coords(-labelpad, 0.5)

    def set_figurepad(self, figpad):
        """
        Set the padding between the left and bottom sides of the figure if the
        labels are being cut off.
        """
        self.cornerfig.subplots_adjust(bottom=figpad, left=figpad)

    def set_yticks(self, tag, ticks, ticklabels=None, **kwargs):
        """
        Set the yticks and ticklabels of the parameter corresponding to 'tag'.
        Any kwargs provided are passed to matplotlib.axes.Axes.set_yticks.
        See the documentation of this method for more information.

        If ticks='plain' is passed, calls 
            Axes.ticklabel_format(style='plain', useOffset=False)
        to remove scientific notation.
        """
        if tag not in list(self.tags):
            raise ValueError(f"'{tag}' is not a valid tag.")
        try:
            ylabel_axes = self._get_row_axes(tag)[0]
        except ValueError:
            raise ValueError(f"'{tag}' has no y-axis.")
        if ticks == 'plain':
            ylabel_axes.ticklabel_format(axis='y', style='plain', useOffset=False)
        else:
            ylabel_axes.set_yticks(ticks, ticklabels, **kwargs)

    def set_xticks(self, tag, ticks, ticklabels=None, **kwargs):
        """
        Set the xticks and ticklabels of the parameter corresponding to 'tag'.
        Any kwargs provided are passed to matplotlib.axes.Axes.set_xticks.
        See the documentation of this method for more information.
        
        If ticks='plain' is passed, calls 
            Axes.ticklabel_format(style='plain', useOffset=False)
        to remove scientific notation.
        """
        if tag not in list(self.tags):
            raise ValueError(f"'{tag}' is not a valid tag.")
        
        try:
            xlabel_axes = self._get_column_axes(tag)[-1]
        except ValueError:
            # The last tag in the tag list is a histogram axes - special case.
            if tag == list(self.tags)[-1]:
                xlabel_axes = self._get_hist_axes(tag)
        if ticks == 'plain':
            xlabel_axes.ticklabel_format(axis='x', style='plain', useOffset=False)
        else:
            xlabel_axes.set_xticks(ticks, ticklabels, **kwargs)
    
    def set_label_sizes(self, labelsize):
        """
        (Re)set the font size of the labels in the corner plot.
        """
        for tag, label in zip(self.tags, self.labels):
            self.set_label(tag, label, fontsize=labelsize)
    
    def set_label(self, tag, label, **kwargs):
        """
        (Re)set the plotting label of the parameter corresponding to 'tag'. Any
        keyword arguments are passed to the set_xlabel and set_ylabel methods 
        of matplotlib.axes.Axes.
        See the documentation of either of these methods for more information.

        Returns dict with keys 'xlabel' and 'ylabel' with values of the label
        successfully set. If a label was not set successfully (e.g. if setting
        the ylabel of the first parameter in the corner plot), the value will be
        None.
        """
        if tag not in list(self.tags):
            raise ValueError(f"'{tag}' is not a valid tag.")
        
        label_dict = {'xlabel':None, 'ylabel':None}
        try:
            ylabel_axes = self._get_row_axes(tag)[0]
            ylabel_axes.set_ylabel(label, **kwargs)
            label_dict['xlabel'] = label
        except ValueError:
            pass
        try:
            xlabel_axes = self._get_column_axes(tag)[-1]
            xlabel_axes.set_xlabel(label, **kwargs)
            label_dict['ylabel'] = label
        except ValueError:
            # The last tag in the tag list is a histogram axes - special case.
            if tag == list(self.tags)[-1]:
                ylabel_axes = self._get_hist_axes(tag)
                ylabel_axes.set_xlabel(label, **kwargs)
                label_dict['xlabel'] = label
                
        # Update the label attribute with the updated label if the re-labelling
        # was successful.
        if label_dict['xlabel'] is not None or label_dict['ylabel'] is not None:
            tag_index = list(self.tags).index(tag)
            self.labels[tag_index] = label
        return label_dict

    def set_lim(self, tag, lower=None, upper=None, **kwargs):
        """
        Set the plotting limits of the parameter corresponding to 'tag'. Any
        keyword arguments are passed to the set_xlim and set_ylim methods of
        matplotlib.axes.Axes.
        See the documentation of either of these methods for more information.

        Returns dict with keys 'xlim' and 'ylim' with values of the limits
        successfully set. If a limit was not set successfully (e.g. if setting
        the ylim of the first parameter in the corner plot), the value will be
        None.
        """
        if tag not in list(self.tags):
            raise ValueError(f"'{tag}' is not a valid tag.")
        
        lim_dict = {'xlim':None, 'ylim':None}
        try:
            row_axes = self._get_row_axes(tag)
            for axes in row_axes:
                axes.set_ylim(bottom=lower, top=upper, **kwargs)
            lim_dict['ylim'] = (lower, upper)
        except ValueError:
            pass
        try:
            column_axes = self._get_column_axes(tag)
            for axes in column_axes:
                axes.set_xlim(left=lower, right=upper, **kwargs)
            lim_dict['xlim'] = (lower, upper)
        except ValueError:
            pass
        
        # All parameters have a hist axis.
        hist_axes = self._get_hist_axes(tag)
        hist_axes.set_xlim(left=lower, right=upper, **kwargs)

    def get_figure(self):
        """
        Return the corner plot.
        """
        return self.cornerfig
