document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
    if (form) { // Chỉ chạy nếu form tồn tại (trên trang predict.html)
        const inputs = document.querySelectorAll('input[type="number"]');
        const button = document.querySelector('button[type="submit"]');
        const buttonError = document.createElement('div');
        buttonError.className = 'button-error';
        button.parentNode.insertBefore(buttonError, button.nextSibling);

        function checkInputs(setInvalidBorder = false) {
            let hasError = false;
            inputs.forEach(input => {
                const value = input.value.trim();
                let errorMessage = '';

                if (value === '') {
                    errorMessage = 'Please enter a number.';
                } else if (!/^-?\d*\.?\d*$/.test(value)) {
                    errorMessage = 'Please enter only numbers (decimal point allowed).';
                } else if (parseFloat(value) < 0) {
                    errorMessage = 'Please enter a non-negative value.';
                }

                const errorDiv = input.parentNode.querySelector('.field-error');
                if (setInvalidBorder) {
                    errorDiv.textContent = errorMessage;
                    if (errorMessage) {
                        input.classList.add('invalid');
                    } else {
                        input.classList.remove('invalid');
                    }
                }

                if (errorMessage) hasError = true;
            });

            return !hasError;
        }

        inputs.forEach(input => {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'field-error';
            input.parentNode.appendChild(errorDiv);
            errorDiv.textContent = '';
            input.classList.remove('invalid');

            input.addEventListener('input', () => {
                const value = input.value.trim();
                let errorMessage = '';

                if (value !== '' && !/^-?\d*\.?\d*$/.test(value)) {
                    errorMessage = 'Please enter only numbers (decimal point allowed).';
                } else if (value !== '' && parseFloat(value) < 0) {
                    errorMessage = 'Please enter a non-negative value.';
                } else if (value === '') {
                    errorMessage = 'Please enter a number.';
                }

                input.parentNode.querySelector('.field-error').textContent = errorMessage;
                input.classList.remove('invalid');
                checkInputs();
                buttonError.textContent = '';
            });
        });

        button.addEventListener('mouseover', () => checkInputs(false));

        button.addEventListener('click', (event) => {
            if (!checkInputs(true)) {
                event.preventDefault();
                buttonError.textContent = 'Please enter all required information.';
            } else {
                buttonError.textContent = '';
                inputs.forEach(input => {
                    input.classList.remove('invalid');
                    input.parentNode.querySelector('.field-error').textContent = '';
                });
            }
        });

        checkInputs(false);
    }
});