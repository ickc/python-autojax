SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = dist/docs

# for bump-my-version, valid options are: major, minor, patch
PART ?= patch

PORT ?= 8098
DOC_DEP = $(shell find docs -type f \( -name '*.md' -o -name '*.rst' \)) $(shell find src -type f -name '*.py')

# documentation ################################################################

.PHONY: all doc epub
all: doc epub pdf man txt html
doc: $(BUILDDIR)/dirhtml/.sentinel
epub: $(BUILDDIR)/epub/SOUKDataCentre.epub
pdf: $(BUILDDIR)/latexpdf/latex/soukdatacentre.pdf
man: $(BUILDDIR)/man/soukdatacentre.1
txt: $(BUILDDIR)/singlehtml/soukdatacentre.txt
html: $(BUILDDIR)/singlehtml/index.html

$(BUILDDIR)/dirhtml/.sentinel: $(DOC_DEP)
	@$(SPHINXBUILD) -b dirhtml "$(SOURCEDIR)" "$(BUILDDIR)/dirhtml" $(SPHINXOPTS)
	touch $@
$(BUILDDIR)/epub/SOUKDataCentre.epub: $(DOC_DEP)
	@$(SPHINXBUILD) -b epub "$(SOURCEDIR)" "$(BUILDDIR)/epub" $(SPHINXOPTS)
$(BUILDDIR)/latexpdf/latex/soukdatacentre.pdf: $(DOC_DEP)
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)/latexpdf" $(SPHINXOPTS)
$(BUILDDIR)/man/soukdatacentre.1: $(DOC_DEP)
	@$(SPHINXBUILD) -b man "$(SOURCEDIR)" "$(BUILDDIR)/man" $(SPHINXOPTS)
$(BUILDDIR)/singlehtml/index.html: $(DOC_DEP)
	@$(SPHINXBUILD) -b singlehtml "$(SOURCEDIR)" "$(BUILDDIR)/singlehtml" $(SPHINXOPTS)
$(BUILDDIR)/singlehtml/soukdatacentre.txt: $(BUILDDIR)/singlehtml/index.html
	pandoc -f html -t plain $< -o $@

.PHONY: serve
serve: doc
	sphinx-autobuild \
		-b dirhtml $(SPHINXOPTS) \
		--port $(PORT) \
		--open-browser \
		--delay 0 \
		"$(SOURCEDIR)" "$(BUILDDIR)"

# testing ######################################################################

test:
	python \
		-m coverage run \
		-m pytest -vv $(PYTESTARGS)

coverage: test
	coverage report

# releasing ####################################################################

.PHONY: bump linkcheck
bump:
	bump-my-version bump $(PART)
	git push --follow-tags

linkcheck:
	linkcheck --external --no-check-anchors --skip-file docs/.skip.linkcheck https://ickc.github.io/python-autojax/

linkcheck-local:
	linkcheck --external --no-check-anchors --skip-file docs/.skip.linkcheck http://127.0.0.1:$(PORT)

################################################################################

.PHONY: format format-py

format: format-py
format-py:
	autoflake --in-place --recursive --expand-star-imports --remove-all-unused-imports --ignore-init-module-imports --remove-duplicate-keys --remove-unused-variables .
	black .
	isort .

.PHONY: clean
clean:
	rm -rf dist

print-%:
	$(info $* = $($*))
