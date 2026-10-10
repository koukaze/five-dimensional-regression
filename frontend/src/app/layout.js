import Link from 'next/link';
import './globals.css';

export const metadata = {
    title: '5D Regression Interpolator',
    description: 'Train and evaluate a multilayer perceptron for five-dimensional regression.',
};

export default function RootLayout({ children }) {
    return (
        <html lang="en">
            <body>
                <a className="skip-link" href="#main-content">Skip to content</a>
                <header className="header">
                    <Link className="brand" href="/">5D <span>Regression</span></Link>
                    <nav aria-label="Main navigation">
                        <Link href="/upload">Upload</Link>
                        <Link href="/train">Train</Link>
                        <Link href="/predict">Predict</Link>
                    </nav>
                </header>
                {children}
                <footer>Five inputs. One continuous target.</footer>
            </body>
        </html>
    );
}
