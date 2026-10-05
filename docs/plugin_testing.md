# Testing the Claude Code plugin locally

You need the `claude` CLI and `pip install "stepscribe-cad[mcp]"` (or `pip install -e ".[dev,mcp]"` in a checkout).

1. Validate the files:

   ```bash
   claude plugin validate plugins/stepscribe --strict
   claude plugin validate . --strict
   ```

2. Add the marketplace from this checkout and install the plugin:

   ```bash
   claude plugin marketplace add .
   claude plugin install stepscribe@stepscribe
   claude plugin list
   ```

3. Start a session in a folder that holds a STEP file (the repository has small ones under `tests/fixtures/step/` after the tests have run once) and try each command:

   ```text
   /stepscribe:export tests/fixtures/step/two_plates_assembly.step
   /stepscribe:interview tests/fixtures/step/understanding/two_link_arm.step
   /stepscribe:review tests/fixtures/step/understanding/two_link_arm.step
   ```

4. Check, for each command:
   - the ETA is announced and progress lines appear while the job runs;
   - the files `*_FULL.md`, `*_SMALL.md`, the folder and the zip are reported;
   - the interview asks one question per message and understands "skip", "not sure", "back" and "done";
   - the review has the sections Summary, Critical issues, Important, Minor, Questions for the designer and What is done well, and cites IDs;
   - **nothing opens a browser window and no local server starts** at any point.

5. Remove it again: `claude plugin uninstall stepscribe@stepscribe` and `claude plugin marketplace remove stepscribe`.

The automated checks that back this up are in `tests/test_plugin.py` (formats, version sync, a scripted MCP session with server start and browser opening forbidden, and a launch of the `.mcp.json` command).
