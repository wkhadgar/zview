# ZView Contribution Guidelines

ZView is an open-source `west` extension designed for high-fidelity Zephyr RTOS runtime visualization. To maintain technical clarity and system stability as the project grows, all contributions must adhere to the following standards.

* **Licensing**: ZView is licensed under the **Apache 2.0 license**. All contributions must be compatible with this license.

* **Developer Certificate of Origin (DCO)**: The project follows the DCO process. Every commit must include a ``Signed-off-by: Name <email>`` line to verify licensing compliance.

* **Technical Verification**: Submissions must demonstrate resilience against asynchronous hardware events.
    * Logic changes must be validated against MCU state transitions and memory rollovers.
    * UI components must maintain visual integrity across varying terminal geometries.

* **Continuous Integration (CI)**: PRs are subject to automated checks for Git formatting, coding style (linting), and functional integrity.

* **Issue Tracking**: Use GitHub issues for bug reports and feature proposals.

* **Environment**: Development is supported on Linux (preferred), macOS, and Windows.

* **Community & Support**: For architectural questions or assistance, refer to the project repository or open an issue with the `question` tag.
