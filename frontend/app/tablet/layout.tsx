import type { Metadata, Viewport } from "next";

export const metadata: Metadata = {
  title: "Nextis",
  description: "Robot cell control panel",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Nextis",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: "cover",
  themeColor: "#000000",
};

export default function TabletLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <div className="dark">{children}</div>;
}
